from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Dict, Mapping, Sequence

import numpy as np
import openmm as mm
from openmm import app as mmapp
from openmm import unit
from rdkit import Chem

from poly_csp.config.schema import MonomerRepresentation, PolymerKind, Site
from poly_csp.forcefield.gaff import build_fragment_prmtop, parameterize_gaff_fragment
from poly_csp.topology.atom_mapping import attachment_instance_maps
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.topology.utils import residue_label_maps


@dataclass(frozen=True)
class ConnectorParams:
    polymer: PolymerKind | None = None
    selector_name: str | None = None
    site: Site | None = None
    monomer_representation: MonomerRepresentation | None = None
    bond_params: Dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    angle_params: Dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    torsion_params: tuple[tuple[tuple[str, str, str, str], tuple[int, float, float]], ...] = ()
    connector_atom_roles: Dict[str, str] = field(default_factory=dict)
    source_prmtop: str | None = None
    fragment_atom_count: int | None = None


@dataclass(frozen=True)
class CappedMonomerFragment:
    mol: Chem.Mol
    atom_roles: Dict[str, int] = field(default_factory=dict)
    connector_roles: Dict[str, int] = field(default_factory=dict)
    connector_atom_roles: Dict[str, str] = field(default_factory=dict)


def build_capped_monomer_fragment(
    polymer: PolymerKind,
    selector_template: SelectorTemplate,
    site: Site,
    monomer_representation: MonomerRepresentation = "natural_oh",
) -> CappedMonomerFragment:
    """Build a single-residue capped fragment with one attached selector.

    The fragment is the topology/structure precursor for capped-monomer
    connector parameter extraction. Atom roles are assigned in semantic
    space:
    - backbone atoms: ``BB_<label>`` from residue label metadata
    - attached selector atoms: ``SL_<local_idx>``
    """
    template = make_glucose_template(
        polymer,
        monomer_representation=monomer_representation,
    )
    frag = polymerize(
        template=template,
        dp=1,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    coords = np.asarray(
        template.mol.GetConformer(0).GetPositions(), dtype=float
    ).reshape((-1, 3))
    frag = assign_conformer(frag, coords)
    frag = attach_selector(
        mol_polymer=frag,
        template=template,
        residue_index=0,
        site=site,
        selector=selector_template,
        linkage_type=selector_template.linkage_type,
    )

    atom_roles: Dict[str, int] = {}
    connector_roles: Dict[str, int] = {}

    label_map = residue_label_maps(frag)[0]
    for label, atom_idx in label_map.items():
        role = f"BB_{label}"
        atom_roles[role] = int(atom_idx)
        frag.GetAtomWithIdx(int(atom_idx)).SetProp("_poly_csp_fragment_role", role)

    attachment_maps = attachment_instance_maps(frag)
    if not attachment_maps:
        raise ValueError("Attached selector fragment is missing instance metadata.")
    if len(attachment_maps) != 1:
        raise ValueError(f"Expected exactly one selector instance, got {len(attachment_maps)}.")
    instance_map = next(iter(attachment_maps.values()))
    for local_idx, atom_idx in instance_map.items():
        role = f"SL_{local_idx:03d}"
        atom_roles[role] = int(atom_idx)
        frag.GetAtomWithIdx(int(atom_idx)).SetProp("_poly_csp_fragment_role", role)

    for local_idx, role_name in selector_template.connector_local_roles.items():
        if local_idx not in instance_map:
            raise ValueError(
                f"Connector local index {local_idx} is missing from attached fragment."
            )
        connector_roles[role_name] = int(instance_map[local_idx])

    connector_atom_roles = {
        role_name: f"SL_{local_idx:03d}"
        for local_idx, role_name in selector_template.connector_local_roles.items()
    }

    return CappedMonomerFragment(
        mol=frag,
        atom_roles=atom_roles,
        connector_roles=connector_roles,
        connector_atom_roles=connector_atom_roles,
    )


def _canonical_bond_roles(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _canonical_angle_roles(a: str, b: str, c: str) -> tuple[str, str, str]:
    return (a, b, c) if a <= c else (c, b, a)


def extract_linkage_params_from_system(
    ref_system: mm.System,
    fragment: CappedMonomerFragment,
    source_prmtop: str | None = None,
) -> ConnectorParams:
    """Extract connector-specific bonded terms from a reference fragment system."""
    idx_to_role = {idx: role for role, idx in fragment.atom_roles.items()}
    connector_atom_roles = set(fragment.connector_atom_roles.values())

    bond_params: Dict[tuple[str, str], tuple[float, float]] = {}
    angle_params: Dict[tuple[str, str, str], tuple[float, float]] = {}
    torsion_terms: list[tuple[tuple[str, str, str, str], tuple[int, float, float]]] = []

    for force_idx in range(ref_system.getNumForces()):
        force = ref_system.getForce(force_idx)
        if isinstance(force, mm.HarmonicBondForce):
            for bond_idx in range(force.getNumBonds()):
                a, b, r0, k = force.getBondParameters(bond_idx)
                roles = (idx_to_role.get(int(a)), idx_to_role.get(int(b)))
                if any(role is None for role in roles):
                    continue
                if not connector_atom_roles.intersection(roles):
                    continue
                bond_params[_canonical_bond_roles(roles[0], roles[1])] = (
                    float(r0.value_in_unit(unit.nanometer)),
                    float(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)),
                )

        elif isinstance(force, mm.HarmonicAngleForce):
            for angle_idx in range(force.getNumAngles()):
                a, b, c, theta0, k = force.getAngleParameters(angle_idx)
                roles = (
                    idx_to_role.get(int(a)),
                    idx_to_role.get(int(b)),
                    idx_to_role.get(int(c)),
                )
                if any(role is None for role in roles):
                    continue
                if not connector_atom_roles.intersection(roles):
                    continue
                angle_params[_canonical_angle_roles(roles[0], roles[1], roles[2])] = (
                    float(theta0.value_in_unit(unit.radian)),
                    float(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)),
                )

        elif isinstance(force, mm.PeriodicTorsionForce):
            for torsion_idx in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, k = force.getTorsionParameters(torsion_idx)
                roles = (
                    idx_to_role.get(int(a)),
                    idx_to_role.get(int(b)),
                    idx_to_role.get(int(c)),
                    idx_to_role.get(int(d)),
                )
                if any(role is None for role in roles):
                    continue
                if not connector_atom_roles.intersection(roles):
                    continue
                torsion_terms.append(
                    (
                        (roles[0], roles[1], roles[2], roles[3]),
                        (
                            int(periodicity),
                            float(phase.value_in_unit(unit.radian)),
                            float(k.value_in_unit(unit.kilojoule_per_mole)),
                        ),
                    )
                )

    selector_name = None
    if fragment.mol.HasProp("_poly_csp_selector_count") and fragment.mol.GetIntProp("_poly_csp_selector_count") > 0:
        for atom in fragment.mol.GetAtoms():
            if atom.HasProp("_poly_csp_component") and atom.GetProp("_poly_csp_component") in {"selector", "connector"}:
                selector_name = "attached_selector"
                break

    return ConnectorParams(
        selector_name=selector_name,
        bond_params=bond_params,
        angle_params=angle_params,
        torsion_params=tuple(torsion_terms),
        connector_atom_roles=dict(fragment.connector_atom_roles),
        source_prmtop=source_prmtop,
        fragment_atom_count=fragment.mol.GetNumAtoms(),
    )


def extract_linkage_params(
    prmtop_path: str | Path,
    fragment: CappedMonomerFragment,
) -> ConnectorParams:
    """Extract connector-specific bonded terms from a capped-fragment prmtop."""
    prmtop = mmapp.AmberPrmtopFile(str(prmtop_path))
    n_top = sum(1 for _ in prmtop.topology.atoms())
    if n_top != fragment.mol.GetNumAtoms():
        raise ValueError(
            f"Fragment/prmtop atom-count mismatch: fragment has {fragment.mol.GetNumAtoms()}, "
            f"prmtop has {n_top}."
        )
    ref_system = prmtop.createSystem()
    return extract_linkage_params_from_system(
        ref_system=ref_system,
        fragment=fragment,
        source_prmtop=str(prmtop_path),
    )


def parameterize_capped_monomer(
    polymer: PolymerKind,
    selector_template: SelectorTemplate,
    site: Site,
    charge_model: str = "bcc",
    net_charge: int = 0,
    monomer_representation: MonomerRepresentation = "natural_oh",
    work_dir: Path | None = None,
) -> ConnectorParams:
    """Parameterize a capped monomer and extract connector bonded terms."""
    if not site:
        raise ValueError("site must be non-empty")
    if selector_template.mol.GetNumAtoms() == 0:
        raise ValueError("selector_template must contain atoms")

    fragment = build_capped_monomer_fragment(
        polymer=polymer,
        selector_template=selector_template,
        site=site,
        monomer_representation=monomer_representation,
    )
    artifacts = parameterize_gaff_fragment(
        fragment_mol=fragment.mol,
        charge_model=charge_model,
        net_charge=net_charge,
        residue_name="CNN",
        pdb_name="connector_fragment.pdb",
        mol2_name="connector_fragment.mol2",
        frcmod_name="connector_fragment.frcmod",
        lib_name="connector_fragment.lib",
        work_dir=work_dir,
    )
    prmtop_path = build_fragment_prmtop(
        mol2_path=artifacts["mol2"],
        frcmod_path=artifacts["frcmod"],
        prmtop_name="connector_fragment.prmtop",
        inpcrd_name="connector_fragment.inpcrd",
        clean_mol2_name="connector_fragment_clean.mol2",
        work_dir=work_dir,
    )
    params = extract_linkage_params(prmtop_path=prmtop_path, fragment=fragment)
    return ConnectorParams(
        polymer=polymer,
        selector_name=selector_template.name,
        site=site,
        monomer_representation=monomer_representation,
        bond_params=params.bond_params,
        angle_params=params.angle_params,
        torsion_params=params.torsion_params,
        connector_atom_roles=params.connector_atom_roles,
        source_prmtop=params.source_prmtop,
        fragment_atom_count=params.fragment_atom_count,
    )
