"""Connector payload extraction from complete capped monomer references.

This module has one canonical job: build a chemically complete selector-bearing
monomer fragment, parameterize that whole fragment, then partition out the
connector-owned runtime payload. There is intentionally no truncated-connector
fallback path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Mapping, Sequence

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
ConnectorLinkageType = Literal["carbamate", "ester", "ether"]
_BACKBONE_ANCHOR_BY_SITE: dict[Site, str] = {
    "C2": "O2",
    "C3": "O3",
    "C6": "O6",
}
_REQUIRED_CONNECTOR_ROLES: dict[ConnectorLinkageType, tuple[str, ...]] = {
    "carbamate": ("amide_n", "carbonyl_c", "carbonyl_o"),
    "ester": ("carbonyl_c", "carbonyl_o"),
    "ether": (),
}


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
    linkage_type: ConnectorLinkageType | None = None
    atom_params: dict[str, ConnectorAtomParams] = field(default_factory=dict)
    connector_role_atom_names: dict[str, str] = field(default_factory=dict)
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


@dataclass(frozen=True)
class ConnectorReferenceMetadata:
    forcefield_mol: Chem.Mol
    source_by_name: dict[str, str]
    atom_name_map: dict[int, str]
    connector_role_atom_names: dict[str, str]
    allowed_names_by_source: dict[str, frozenset[str]]


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


def _forcefield_fragment_metadata(fragment: CappedMonomerFragment) -> ConnectorReferenceMetadata:
    forcefield = build_forcefield_molecule(fragment.mol)
    atom_names = {
        int(entry.atom_index): str(entry.atom_name)
        for entry in forcefield.manifest
    }
    source_by_name = {
        str(entry.atom_name): str(entry.source)
        for entry in forcefield.manifest
    }
    fragment_role_to_atom_name = {
        atom.GetProp("_poly_csp_fragment_role"): atom.GetProp("_poly_csp_atom_name")
        for atom in forcefield.mol.GetAtoms()
        if atom.HasProp("_poly_csp_fragment_role") and atom.HasProp("_poly_csp_atom_name")
    }
    connector_role_atom_names = {
        role_name: fragment_role_to_atom_name[fragment_role]
        for role_name, fragment_role in fragment.connector_atom_roles.items()
        if fragment_role in fragment_role_to_atom_name
    }
    allowed_names_by_source: dict[str, frozenset[str]] = {
        source: frozenset(
            name
            for name, observed_source in source_by_name.items()
            if observed_source == source
        )
        for source in {"backbone", "selector", "connector"}
    }
    return ConnectorReferenceMetadata(
        forcefield_mol=forcefield.mol,
        source_by_name=source_by_name,
        atom_name_map=atom_names,
        connector_role_atom_names=connector_role_atom_names,
        allowed_names_by_source=allowed_names_by_source,
    )


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


def _touches_connector_tokens(tokens: Sequence[ConnectorToken]) -> bool:
    return any(token.source == "connector" for token in tokens)


def _token_sources(tokens: Sequence[ConnectorToken]) -> set[str]:
    return {str(token.source) for token in tokens}


def _torsion_contains_connector_pair(
    template: ConnectorTorsionTemplate,
    atom_names: set[str],
) -> bool:
    connector_names = {
        token.atom_name
        for token in template.atoms
        if token.source == "connector"
    }
    return atom_names.issubset(connector_names)


def _validate_connector_planarity_terms(params: ConnectorParams) -> None:
    if params.linkage_type is None or params.site is None:
        raise ValueError(
            "Connector payload is missing linkage/site metadata for planarity validation."
        )
    anchor = _BACKBONE_ANCHOR_BY_SITE[params.site]
    role_names = dict(params.connector_role_atom_names)
    torsions = tuple(params.torsions)

    if params.linkage_type == "carbamate":
        carbonyl_c = role_names["carbonyl_c"]
        carbonyl_o = role_names["carbonyl_o"]
        amide_n = role_names["amide_n"]

        if not any(
            _torsion_contains_connector_pair(template, {carbonyl_c, amide_n})
            and "selector" in _token_sources(template.atoms)
            for template in torsions
        ):
            raise ValueError(
                "Carbamate connector payload is missing a selector-facing torsion "
                "that preserves carbonyl_c/amide_n planarity."
            )
        if not any(
            _torsion_contains_connector_pair(template, {carbonyl_c, amide_n})
            and any(
                token.source == "backbone" and token.atom_name == anchor
                for token in template.atoms
            )
            for template in torsions
        ):
            raise ValueError(
                "Carbamate connector payload is missing a backbone-anchor torsion "
                f"through {anchor} and the carbonyl_c/amide_n pair."
            )
        if not any(
            _torsion_contains_connector_pair(template, {carbonyl_c, carbonyl_o})
            and "selector" in _token_sources(template.atoms)
            for template in torsions
        ):
            raise ValueError(
                "Carbamate connector payload is missing a selector-facing torsion "
                "that preserves carbonyl_o/carbonyl_c planarity."
            )
        return

    if params.linkage_type == "ester":
        carbonyl_c = role_names["carbonyl_c"]
        carbonyl_o = role_names["carbonyl_o"]

        if not any(
            _torsion_contains_connector_pair(template, {carbonyl_c, carbonyl_o})
            and "selector" in _token_sources(template.atoms)
            for template in torsions
        ):
            raise ValueError(
                "Ester connector payload is missing a selector-facing torsion "
                "that preserves carbonyl_o/carbonyl_c planarity."
            )
        if not any(
            carbonyl_c
            in {
                token.atom_name
                for token in template.atoms
                if token.source == "connector"
            }
            and any(
                token.source == "backbone" and token.atom_name == anchor
                for token in template.atoms
            )
            and "selector" in _token_sources(template.atoms)
            for template in torsions
        ):
            raise ValueError(
                "Ester connector payload is missing a backbone-anchor torsion "
                f"through {anchor} and the carbonyl_c selector boundary."
            )


def validate_connector_params(
    params: ConnectorParams,
    *,
    allowed_names_by_source: Mapping[str, frozenset[str]] | None = None,
) -> None:
    """Validate the canonical connector ownership contract and planarity terms."""
    if not params.atom_params:
        raise ValueError("Connector payload extraction produced no connector atom parameters.")
    if params.linkage_type is None:
        raise ValueError("Connector payload is missing linkage_type metadata.")

    required_roles = _REQUIRED_CONNECTOR_ROLES[params.linkage_type]
    observed_roles = tuple(sorted(params.connector_role_atom_names))
    if tuple(sorted(required_roles)) != observed_roles:
        raise ValueError(
            "Connector payload role map does not match the required connector roles "
            f"for linkage_type={params.linkage_type!r}. "
            f"Expected={sorted(required_roles)}, observed={list(observed_roles)}."
        )

    connector_names = set(params.atom_params)
    if not set(params.connector_role_atom_names.values()).issubset(connector_names):
        raise ValueError("Connector role map references atoms that are not connector-owned.")

    if allowed_names_by_source is not None:
        for source in ("backbone", "selector", "connector"):
            if source not in allowed_names_by_source:
                raise ValueError(
                    "Connector payload validation is missing the allowed-name set "
                    f"for source {source!r}."
                )

    for templates, label in (
        (params.bonds, "bond"),
        (params.angles, "angle"),
        (params.torsions, "torsion"),
    ):
        for template in templates:
            tokens = template.atoms
            if not _touches_connector_tokens(tokens):
                raise ValueError(
                    f"Connector {label} payload contains a term with no connector atoms."
                )
            for token in tokens:
                if token.source == "connector" and token.atom_name not in connector_names:
                    raise ValueError(
                        f"Connector {label} payload references unknown connector atom {token.atom_name!r}."
                    )
                if (
                    allowed_names_by_source is not None
                    and token.atom_name not in allowed_names_by_source[token.source]
                ):
                    raise ValueError(
                        "Connector "
                        f"{label} payload references an atom outside its allowed {token.source!r} "
                        f"name set: {token.atom_name!r}."
                    )

    _validate_connector_planarity_terms(params)


def _infer_fragment_linkage_type(fragment: CappedMonomerFragment) -> ConnectorLinkageType:
    role_names = set(fragment.connector_roles)
    if "amide_n" in role_names:
        return "carbamate"
    if "carbonyl_c" in role_names and "carbonyl_o" in role_names:
        return "ester"
    return "ether"


def _infer_fragment_site(fragment: CappedMonomerFragment) -> Site:
    connector_indices = set(fragment.connector_roles.values())
    for site, anchor in _BACKBONE_ANCHOR_BY_SITE.items():
        role = f"BB_{anchor}"
        atom_idx = fragment.atom_roles.get(role)
        if atom_idx is None:
            continue
        atom = fragment.mol.GetAtomWithIdx(int(atom_idx))
        if any(int(neighbor.GetIdx()) in connector_indices for neighbor in atom.GetNeighbors()):
            return site
    raise ValueError("Could not infer connector attachment site from the capped monomer fragment.")


def _finalize_connector_params(
    *,
    polymer: PolymerKind | None,
    selector_name: str | None,
    site: Site | None,
    monomer_representation: MonomerRepresentation | None,
    linkage_type: ConnectorLinkageType | None,
    connector_role_atom_names: dict[str, str],
    atom_params: dict[str, ConnectorAtomParams],
    bonds: dict[tuple[ConnectorToken, ConnectorToken], ConnectorBondTemplate],
    angles: dict[tuple[ConnectorToken, ConnectorToken, ConnectorToken], ConnectorAngleTemplate],
    torsions: list[ConnectorTorsionTemplate],
    source_prmtop: str | None,
    fragment_atom_count: int | None,
    allowed_names_by_source: Mapping[str, frozenset[str]] | None = None,
) -> ConnectorParams:
    params = ConnectorParams(
        polymer=polymer,
        selector_name=selector_name,
        site=site,
        monomer_representation=monomer_representation,
        linkage_type=linkage_type,
        atom_params=atom_params,
        connector_role_atom_names=dict(connector_role_atom_names),
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
        fragment_atom_count=fragment_atom_count,
    )
    validate_connector_params(
        params,
        allowed_names_by_source=allowed_names_by_source,
    )
    return params


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
    metadata = _forcefield_fragment_metadata(fragment)
    forcefield_mol = metadata.forcefield_mol
    source_by_name = metadata.source_by_name
    atom_name_map = dict(metadata.atom_name_map)

    artifacts = parameterize_gaff_fragment(
        fragment_mol=forcefield_mol,
        charge_model=charge_model,
        net_charge=net_charge,
        residue_name="CNN",
        input_name="connector_fragment.mol",
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

    return _finalize_connector_params(
        polymer=polymer,
        selector_name=selector_template.name,
        site=site,
        monomer_representation=monomer_representation,
        linkage_type=selector_template.linkage_type,
        connector_role_atom_names=metadata.connector_role_atom_names,
        atom_params=atom_params,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        source_prmtop=str(prmtop_path),
        fragment_atom_count=forcefield_mol.GetNumAtoms(),
        allowed_names_by_source=metadata.allowed_names_by_source,
    )


def extract_linkage_params_from_system(
    ref_system: mm.System,
    fragment: CappedMonomerFragment,
    source_prmtop: str | None = None,
) -> ConnectorParams:
    """Test helper: extract connector payloads from a prebuilt reference system."""
    metadata = _forcefield_fragment_metadata(fragment)
    forcefield_mol = metadata.forcefield_mol
    source_by_name = metadata.source_by_name
    idx_to_name = dict(metadata.atom_name_map)

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

    return _finalize_connector_params(
        polymer=None,
        selector_name="attached_selector",
        site=_infer_fragment_site(fragment),
        monomer_representation=None,
        linkage_type=_infer_fragment_linkage_type(fragment),
        connector_role_atom_names=metadata.connector_role_atom_names,
        atom_params=atom_params,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        source_prmtop=source_prmtop,
        fragment_atom_count=forcefield_mol.GetNumAtoms(),
        allowed_names_by_source=metadata.allowed_names_by_source,
    )
