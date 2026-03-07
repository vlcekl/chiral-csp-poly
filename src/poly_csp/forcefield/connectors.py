from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Sequence

import openmm as mm
from openmm import app as mmapp
from openmm import unit
from rdkit import Chem

from poly_csp.config.schema import HelixSpec, MonomerRepresentation, PolymerKind, Site
from poly_csp.forcefield.gaff import build_fragment_prmtop, parameterize_gaff_fragment
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.atom_mapping import attachment_instance_maps
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.topology.utils import residue_label_maps


ConnectorSource = Literal["backbone", "selector", "connector"]


@dataclass(frozen=True)
class ConnectorToken:
    source: ConnectorSource
    atom_name: str


@dataclass(frozen=True)
class ConnectorAtomParams:
    atom_name: str
    charge_e: float
    sigma_nm: float
    epsilon_kj_per_mol: float


@dataclass(frozen=True)
class ConnectorBondTemplate:
    atoms: tuple[ConnectorToken, ConnectorToken]
    length_nm: float
    k_kj_per_mol_nm2: float


@dataclass(frozen=True)
class ConnectorAngleTemplate:
    atoms: tuple[ConnectorToken, ConnectorToken, ConnectorToken]
    theta0_rad: float
    k_kj_per_mol_rad2: float


@dataclass(frozen=True)
class ConnectorTorsionTemplate:
    atoms: tuple[ConnectorToken, ConnectorToken, ConnectorToken, ConnectorToken]
    periodicity: int
    phase_rad: float
    k_kj_per_mol: float


@dataclass(frozen=True)
class ConnectorParams:
    polymer: PolymerKind | None = None
    selector_name: str | None = None
    site: Site | None = None
    monomer_representation: MonomerRepresentation | None = None
    atom_params: dict[str, ConnectorAtomParams] = field(default_factory=dict)
    bonds: tuple[ConnectorBondTemplate, ...] = ()
    angles: tuple[ConnectorAngleTemplate, ...] = ()
    torsions: tuple[ConnectorTorsionTemplate, ...] = ()
    source_prmtop: str | None = None
    fragment_atom_count: int | None = None


@dataclass(frozen=True)
class CappedMonomerFragment:
    mol: Chem.Mol
    atom_roles: Dict[str, int] = field(default_factory=dict)
    connector_roles: Dict[str, int] = field(default_factory=dict)
    connector_atom_roles: Dict[str, str] = field(default_factory=dict)


def _fragment_atom_role_map(fragment: Chem.Mol) -> dict[int, str]:
    out: dict[int, str] = {}
    for atom in fragment.GetAtoms():
        if atom.HasProp("_poly_csp_manifest_source"):
            source = atom.GetProp("_poly_csp_manifest_source")
            atom_name = atom.GetProp("_poly_csp_atom_name")
            out[int(atom.GetIdx())] = f"{source}:{atom_name}"
    return out


def build_capped_monomer_fragment(
    polymer: PolymerKind,
    selector_template: SelectorTemplate,
    site: Site,
    monomer_representation: MonomerRepresentation = "natural_oh",
) -> CappedMonomerFragment:
    """Build a single-residue capped fragment with one attached selector."""
    template = make_glucose_template(
        polymer,
        monomer_representation=monomer_representation,
    )
    topology = polymerize(
        template=template,
        dp=1,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    frag = build_backbone_structure(
        topology,
        helix_spec=HelixSpec(
            name="fragment",
            theta_rad=0.0,
            rise_A=0.0,
            repeat_residues=1,
            repeat_turns=1,
            residues_per_turn=1.0,
            pitch_A=1.0,
            handedness="right",
        ),
    ).mol
    frag = attach_selector(
        mol_polymer=frag,
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


def _bond_key(
    atoms: tuple[ConnectorToken, ConnectorToken],
) -> tuple[ConnectorToken, ConnectorToken]:
    a, b = atoms
    return (a, b) if (a.source, a.atom_name) <= (b.source, b.atom_name) else (b, a)


def _angle_key(
    atoms: tuple[ConnectorToken, ConnectorToken, ConnectorToken],
) -> tuple[ConnectorToken, ConnectorToken, ConnectorToken]:
    a, b, c = atoms
    return (a, b, c) if (a.source, a.atom_name) <= (c.source, c.atom_name) else (c, b, a)


def _append_unique_torsion(
    bucket: list[ConnectorTorsionTemplate],
    template: ConnectorTorsionTemplate,
) -> None:
    for existing in bucket:
        if existing.atoms != template.atoms:
            continue
        if (
            existing.periodicity == template.periodicity
            and abs(existing.phase_rad - template.phase_rad) <= 1e-8
            and abs(existing.k_kj_per_mol - template.k_kj_per_mol) <= 1e-8
        ):
            return
    bucket.append(template)


def _token_sort_key(token: ConnectorToken) -> tuple[str, str]:
    return (str(token.source), str(token.atom_name))


def _token_tuple_sort_key(tokens: tuple[ConnectorToken, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(_token_sort_key(token) for token in tokens)


def _forcefield_fragment_metadata(fragment: CappedMonomerFragment) -> tuple[Chem.Mol, dict[str, str]]:
    forcefield = build_forcefield_molecule(fragment.mol)
    atom_names = {
        int(entry.atom_index): str(entry.atom_name)
        for entry in forcefield.manifest
    }
    source_by_name = {
        str(entry.atom_name): str(entry.source)
        for entry in forcefield.manifest
    }
    return forcefield.mol, source_by_name


def _tokenize_terms(
    names: Sequence[str],
    source_by_name: dict[str, str],
) -> tuple[ConnectorToken, ...]:
    return tuple(
        ConnectorToken(
            source=str(source_by_name[name]).replace("terminal_cap_left", "backbone").replace("terminal_cap_right", "backbone"),  # type: ignore[arg-type]
            atom_name=str(name),
        )
        for name in names
    )


def load_connector_params(
    polymer: PolymerKind,
    selector_template: SelectorTemplate,
    site: Site,
    charge_model: str = "bcc",
    net_charge: int = 0,
    monomer_representation: MonomerRepresentation = "natural_oh",
    work_dir: Path | None = None,
) -> ConnectorParams:
    """Parameterize a capped monomer and extract connector owned runtime payloads."""
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
    forcefield_mol, source_by_name = _forcefield_fragment_metadata(fragment)
    atom_name_map = {
        int(atom.GetIdx()): atom.GetProp("_poly_csp_atom_name")
        for atom in forcefield_mol.GetAtoms()
    }

    artifacts = parameterize_gaff_fragment(
        fragment_mol=forcefield_mol,
        charge_model=charge_model,
        net_charge=net_charge,
        residue_name="CNN",
        pdb_name="connector_fragment.pdb",
        mol2_name="connector_fragment.mol2",
        frcmod_name="connector_fragment.frcmod",
        lib_name="connector_fragment.lib",
        work_dir=work_dir,
        atom_names=atom_name_map,
    )
    prmtop_path = build_fragment_prmtop(
        mol2_path=artifacts["mol2"],
        frcmod_path=artifacts["frcmod"],
        prmtop_name="connector_fragment.prmtop",
        inpcrd_name="connector_fragment.inpcrd",
        clean_mol2_name="connector_fragment_clean.mol2",
        work_dir=work_dir,
    )

    prmtop = mmapp.AmberPrmtopFile(str(prmtop_path))
    system = prmtop.createSystem()
    prmtop_atoms = list(prmtop.topology.atoms())
    if len(prmtop_atoms) != forcefield_mol.GetNumAtoms():
        raise ValueError(
            "Connector reference atom count changed during AMBER conversion: "
            f"forcefield={forcefield_mol.GetNumAtoms()}, prmtop={len(prmtop_atoms)}."
        )
    prmtop_idx_to_name = {
        int(atom_idx): atom_name_map[int(atom_idx)]
        for atom_idx in range(len(prmtop_atoms))
    }

    nonbonded = None
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if isinstance(force, mm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise ValueError("Connector reference system is missing NonbondedForce.")

    atom_params: dict[str, ConnectorAtomParams] = {}
    for atom_idx, atom_name in prmtop_idx_to_name.items():
        if source_by_name[atom_name] != "connector":
            continue
        charge, sigma, epsilon = nonbonded.getParticleParameters(int(atom_idx))
        atom_params[atom_name] = ConnectorAtomParams(
            atom_name=atom_name,
            charge_e=float(charge.value_in_unit(unit.elementary_charge)),
            sigma_nm=float(sigma.value_in_unit(unit.nanometer)),
            epsilon_kj_per_mol=float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
        )

    bonds: dict[tuple[ConnectorToken, ConnectorToken], ConnectorBondTemplate] = {}
    angles: dict[
        tuple[ConnectorToken, ConnectorToken, ConnectorToken],
        ConnectorAngleTemplate,
    ] = {}
    torsions: list[ConnectorTorsionTemplate] = []

    def _touches_connector(names: Sequence[str]) -> bool:
        return any(source_by_name[name] == "connector" for name in names)

    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)

        if isinstance(force, mm.HarmonicBondForce):
            for bond_idx in range(force.getNumBonds()):
                a, b, r0, k = force.getBondParameters(bond_idx)
                names = (
                    prmtop_idx_to_name.get(int(a)),
                    prmtop_idx_to_name.get(int(b)),
                )
                if any(name is None for name in names):
                    continue
                if not _touches_connector(names):  # type: ignore[arg-type]
                    continue
                tokens = _tokenize_terms((str(names[0]), str(names[1])), source_by_name)
                key = _bond_key((tokens[0], tokens[1]))
                bonds[key] = ConnectorBondTemplate(
                    atoms=key,
                    length_nm=float(r0.value_in_unit(unit.nanometer)),
                    k_kj_per_mol_nm2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                    ),
                )
            continue

        if isinstance(force, mm.HarmonicAngleForce):
            for angle_idx in range(force.getNumAngles()):
                a, b, c, theta0, k = force.getAngleParameters(angle_idx)
                names = (
                    prmtop_idx_to_name.get(int(a)),
                    prmtop_idx_to_name.get(int(b)),
                    prmtop_idx_to_name.get(int(c)),
                )
                if any(name is None for name in names):
                    continue
                if not _touches_connector(names):  # type: ignore[arg-type]
                    continue
                tokens = _tokenize_terms(
                    (str(names[0]), str(names[1]), str(names[2])),
                    source_by_name,
                )
                key = _angle_key((tokens[0], tokens[1], tokens[2]))
                angles[key] = ConnectorAngleTemplate(
                    atoms=key,
                    theta0_rad=float(theta0.value_in_unit(unit.radian)),
                    k_kj_per_mol_rad2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                    ),
                )
            continue

        if isinstance(force, mm.PeriodicTorsionForce):
            for torsion_idx in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, k = force.getTorsionParameters(torsion_idx)
                names = (
                    prmtop_idx_to_name.get(int(a)),
                    prmtop_idx_to_name.get(int(b)),
                    prmtop_idx_to_name.get(int(c)),
                    prmtop_idx_to_name.get(int(d)),
                )
                if any(name is None for name in names):
                    continue
                if not _touches_connector(names):  # type: ignore[arg-type]
                    continue
                tokens = _tokenize_terms(
                    (str(names[0]), str(names[1]), str(names[2]), str(names[3])),
                    source_by_name,
                )
                _append_unique_torsion(
                    torsions,
                    ConnectorTorsionTemplate(
                        atoms=(tokens[0], tokens[1], tokens[2], tokens[3]),
                        periodicity=int(periodicity),
                        phase_rad=float(phase.value_in_unit(unit.radian)),
                        k_kj_per_mol=float(k.value_in_unit(unit.kilojoule_per_mole)),
                    ),
                )

    if not atom_params:
        raise ValueError("Connector payload extraction produced no connector atom parameters.")

    return ConnectorParams(
        polymer=polymer,
        selector_name=selector_template.name,
        site=site,
        monomer_representation=monomer_representation,
        atom_params=atom_params,
        bonds=tuple(sorted(bonds.values(), key=lambda item: _token_tuple_sort_key(item.atoms))),
        angles=tuple(sorted(angles.values(), key=lambda item: _token_tuple_sort_key(item.atoms))),
        torsions=tuple(
            sorted(
                torsions,
                key=lambda item: (
                    _token_tuple_sort_key(item.atoms),
                    item.periodicity,
                    item.phase_rad,
                    item.k_kj_per_mol,
                ),
            )
        ),
        source_prmtop=str(prmtop_path),
        fragment_atom_count=forcefield_mol.GetNumAtoms(),
    )


def extract_linkage_params_from_system(
    ref_system: mm.System,
    fragment: CappedMonomerFragment,
    source_prmtop: str | None = None,
) -> ConnectorParams:
    """Test helper: extract connector payloads from a prebuilt reference system."""
    forcefield_mol, source_by_name = _forcefield_fragment_metadata(fragment)
    atom_name_map = {
        int(atom.GetIdx()): atom.GetProp("_poly_csp_atom_name")
        for atom in forcefield_mol.GetAtoms()
    }
    idx_to_name = atom_name_map

    atom_params: dict[str, ConnectorAtomParams] = {}
    bonds: dict[tuple[ConnectorToken, ConnectorToken], ConnectorBondTemplate] = {}
    angles: dict[
        tuple[ConnectorToken, ConnectorToken, ConnectorToken],
        ConnectorAngleTemplate,
    ] = {}
    torsions: list[ConnectorTorsionTemplate] = []

    for force_idx in range(ref_system.getNumForces()):
        force = ref_system.getForce(force_idx)
        if isinstance(force, mm.NonbondedForce):
            for atom_idx in range(force.getNumParticles()):
                atom_name = idx_to_name.get(int(atom_idx))
                if atom_name is None or source_by_name[atom_name] != "connector":
                    continue
                charge, sigma, epsilon = force.getParticleParameters(int(atom_idx))
                atom_params[atom_name] = ConnectorAtomParams(
                    atom_name=atom_name,
                    charge_e=float(charge.value_in_unit(unit.elementary_charge)),
                    sigma_nm=float(sigma.value_in_unit(unit.nanometer)),
                    epsilon_kj_per_mol=float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
                )
            continue

        if isinstance(force, mm.HarmonicBondForce):
            for bond_idx in range(force.getNumBonds()):
                a, b, r0, k = force.getBondParameters(bond_idx)
                names = (idx_to_name.get(int(a)), idx_to_name.get(int(b)))
                if any(name is None for name in names):
                    continue
                if not any(source_by_name[name] == "connector" for name in names):  # type: ignore[arg-type]
                    continue
                tokens = _tokenize_terms((str(names[0]), str(names[1])), source_by_name)
                key = _bond_key((tokens[0], tokens[1]))
                bonds[key] = ConnectorBondTemplate(
                    atoms=key,
                    length_nm=float(r0.value_in_unit(unit.nanometer)),
                    k_kj_per_mol_nm2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                    ),
                )
            continue

        if isinstance(force, mm.HarmonicAngleForce):
            for angle_idx in range(force.getNumAngles()):
                a, b, c, theta0, k = force.getAngleParameters(angle_idx)
                names = (idx_to_name.get(int(a)), idx_to_name.get(int(b)), idx_to_name.get(int(c)))
                if any(name is None for name in names):
                    continue
                if not any(source_by_name[name] == "connector" for name in names):  # type: ignore[arg-type]
                    continue
                tokens = _tokenize_terms(
                    (str(names[0]), str(names[1]), str(names[2])),
                    source_by_name,
                )
                key = _angle_key((tokens[0], tokens[1], tokens[2]))
                angles[key] = ConnectorAngleTemplate(
                    atoms=key,
                    theta0_rad=float(theta0.value_in_unit(unit.radian)),
                    k_kj_per_mol_rad2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                    ),
                )
            continue

        if isinstance(force, mm.PeriodicTorsionForce):
            for torsion_idx in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, k = force.getTorsionParameters(torsion_idx)
                names = (
                    idx_to_name.get(int(a)),
                    idx_to_name.get(int(b)),
                    idx_to_name.get(int(c)),
                    idx_to_name.get(int(d)),
                )
                if any(name is None for name in names):
                    continue
                if not any(source_by_name[name] == "connector" for name in names):  # type: ignore[arg-type]
                    continue
                tokens = _tokenize_terms(
                    (str(names[0]), str(names[1]), str(names[2]), str(names[3])),
                    source_by_name,
                )
                _append_unique_torsion(
                    torsions,
                    ConnectorTorsionTemplate(
                        atoms=(tokens[0], tokens[1], tokens[2], tokens[3]),
                        periodicity=int(periodicity),
                        phase_rad=float(phase.value_in_unit(unit.radian)),
                        k_kj_per_mol=float(k.value_in_unit(unit.kilojoule_per_mole)),
                    ),
                )

    return ConnectorParams(
        selector_name="attached_selector",
        atom_params=atom_params,
        bonds=tuple(sorted(bonds.values(), key=lambda item: _token_tuple_sort_key(item.atoms))),
        angles=tuple(sorted(angles.values(), key=lambda item: _token_tuple_sort_key(item.atoms))),
        torsions=tuple(
            sorted(
                torsions,
                key=lambda item: (
                    _token_tuple_sort_key(item.atoms),
                    item.periodicity,
                    item.phase_rad,
                    item.k_kj_per_mol,
                ),
            )
        ),
        source_prmtop=source_prmtop,
        fragment_atom_count=forcefield_mol.GetNumAtoms(),
    )
