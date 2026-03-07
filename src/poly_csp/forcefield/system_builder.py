from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from rdkit import Chem

import openmm as mm
from openmm import unit

from poly_csp.forcefield.connectors import (
    ConnectorParams,
    ConnectorToken,
    validate_connector_params,
)
from poly_csp.forcefield.exceptions import apply_mixing_rules
from poly_csp.forcefield.gaff import (
    SelectorAngleTemplate,
    SelectorBondTemplate,
    SelectorFragmentParams,
    SelectorTorsionTemplate,
)
from poly_csp.forcefield.glycam import GlycamParams
from poly_csp.forcefield.glycam_mapping import GlycamMappingResult, map_backbone_to_glycam
from poly_csp.forcefield.selector_mapping import map_selector_instances
from poly_csp.topology.atom_mapping import build_atom_map


@dataclass(frozen=True)
class SystemBuildResult:
    system: mm.System
    positions_nm: unit.Quantity
    excluded_pairs: set[tuple[int, int]]
    nonbonded_mode: str = "soft"
    topology_manifest: tuple[dict[str, object], ...] = ()
    component_counts: dict[str, int] = field(default_factory=dict)
    exception_summary: dict[str, object] = field(default_factory=dict)
    source_manifest: dict[str, object] = field(default_factory=dict)


def _atomic_mass_dalton(z: int) -> float:
    if z <= 1:
        return 1.008
    if z == 6:
        return 12.011
    if z == 7:
        return 14.007
    if z == 8:
        return 15.999
    if z == 16:
        return 32.06
    return 12.0


def _positions_nm_from_mol(mol: Chem.Mol) -> unit.Quantity:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before OpenMM system build.")
    xyz_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    return (xyz_A / 10.0) * unit.nanometer


def exclusion_pairs_from_mol(
    mol: Chem.Mol,
    exclude_13: bool = True,
    exclude_14: bool = False,
) -> set[tuple[int, int]]:
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    max_depth = 1 + int(exclude_13) + int(exclude_14)
    excluded: set[tuple[int, int]] = set()
    for src in range(n):
        q: deque[tuple[int, int]] = deque([(src, 0)])
        seen = {src}
        while q:
            node, depth = q.popleft()
            if depth >= max_depth:
                continue
            for nbr in adj[node]:
                i, j = (src, nbr) if src < nbr else (nbr, src)
                if src != nbr:
                    excluded.add((i, j))
                if nbr not in seen:
                    seen.add(nbr)
                    q.append((nbr, depth + 1))
    return excluded


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _angle_key(a: int, b: int, c: int) -> tuple[int, int, int]:
    return (a, b, c) if a <= c else (c, b, a)


def _torsion_key(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    forward = (a, b, c, d)
    reverse = (d, c, b, a)
    return forward if forward <= reverse else reverse


def _register_term_owner(
    registry: dict[tuple[int, ...], str],
    key: tuple[int, ...],
    owner: str,
    *,
    term_kind: str,
) -> None:
    existing = registry.get(key)
    if existing is None:
        registry[key] = owner
        return
    if existing != owner:
        raise ValueError(
            f"Ambiguous {term_kind} ownership for atoms {key!r}: "
            f"{existing!r} vs {owner!r}."
        )


def _merge_source_manifest(
    base: Mapping[str, object],
    extra: Mapping[str, object] | None,
) -> dict[str, object]:
    if not extra:
        return dict(base)

    def _merge_value(left: object, right: object) -> object:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            merged = {str(key): value for key, value in left.items()}
            for key, value in right.items():
                key_str = str(key)
                if key_str in merged:
                    merged[key_str] = _merge_value(merged[key_str], value)
                else:
                    merged[key_str] = value
            return merged
        return right

    return {
        key: _merge_value(base.get(key), value) if key in base else value
        for key, value in {**dict(base), **dict(extra)}.items()
    }


def _set_or_add_bond(force: mm.HarmonicBondForce, a: int, b: int, r0, k) -> None:
    target = _bond_key(a, b)
    for idx in range(force.getNumBonds()):
        p1, p2, _, _ = force.getBondParameters(idx)
        if _bond_key(int(p1), int(p2)) == target:
            force.setBondParameters(idx, int(p1), int(p2), r0, k)
            return
    force.addBond(int(a), int(b), r0, k)


def _set_or_add_angle(
    force: mm.HarmonicAngleForce,
    a: int,
    b: int,
    c: int,
    theta0,
    k,
) -> None:
    target = _angle_key(a, b, c)
    for idx in range(force.getNumAngles()):
        p1, p2, p3, _, _ = force.getAngleParameters(idx)
        if _angle_key(int(p1), int(p2), int(p3)) == target:
            force.setAngleParameters(idx, int(p1), int(p2), int(p3), theta0, k)
            return
    force.addAngle(int(a), int(b), int(c), theta0, k)


def _component_counts(mol: Chem.Mol) -> dict[str, int]:
    counts = {"backbone": 0, "selector": 0, "connector": 0}
    for atom in mol.GetAtoms():
        source = atom.GetProp("_poly_csp_manifest_source") if atom.HasProp("_poly_csp_manifest_source") else "backbone"
        if source == "selector":
            counts["selector"] += 1
        elif source == "connector":
            counts["connector"] += 1
        elif source == "backbone":
            counts["backbone"] += 1
        elif source.startswith("terminal_cap_"):
            raise ValueError("Canonical runtime system does not support terminal caps yet.")
        else:
            raise ValueError(f"Unsupported manifest source {source!r}.")
    return counts


def _topology_manifest(
    mol: Chem.Mol,
    backbone_mapping: GlycamMappingResult | None = None,
) -> tuple[dict[str, object], ...]:
    glycam_by_atom_index = (
        {int(item.atom_index): item for item in backbone_mapping.assignments}
        if backbone_mapping is not None
        else {}
    )
    out = []
    for atom in mol.GetAtoms():
        entry = {
            "atom_index": int(atom.GetIdx()),
            "atom_name": atom.GetProp("_poly_csp_atom_name") if atom.HasProp("_poly_csp_atom_name") else "",
            "source": atom.GetProp("_poly_csp_manifest_source") if atom.HasProp("_poly_csp_manifest_source") else "backbone",
        }
        if atom.HasProp("_poly_csp_residue_index"):
            entry["residue_index"] = int(atom.GetIntProp("_poly_csp_residue_index"))
        if atom.HasProp("_poly_csp_selector_instance"):
            entry["selector_instance"] = int(atom.GetIntProp("_poly_csp_selector_instance"))
        if atom.HasProp("_poly_csp_selector_name"):
            entry["selector_name"] = atom.GetProp("_poly_csp_selector_name")
        if atom.HasProp("_poly_csp_site"):
            entry["site"] = atom.GetProp("_poly_csp_site")
        mapped = glycam_by_atom_index.get(int(atom.GetIdx()))
        if mapped is not None:
            entry["glycam_residue_role"] = mapped.residue_role
            entry["glycam_residue_name"] = mapped.glycam_residue_name
            entry["glycam_atom_name"] = mapped.glycam_atom_name
        out.append(entry)
    return tuple(out)


def _backbone_name_maps(mol: Chem.Mol) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_manifest_source"):
            continue
        if atom.GetProp("_poly_csp_manifest_source") != "backbone":
            continue
        if not atom.HasProp("_poly_csp_residue_index") or not atom.HasProp("_poly_csp_atom_name"):
            continue
        residue_index = int(atom.GetIntProp("_poly_csp_residue_index"))
        atom_name = atom.GetProp("_poly_csp_atom_name")
        out.setdefault(residue_index, {})[atom_name] = int(atom.GetIdx())
    return out


@dataclass(frozen=True)
class _ConnectorContext:
    instance_id: int
    selector_name: str
    site: str
    residue_index: int
    selector_atoms_by_name: dict[str, int]
    connector_atoms_by_name: dict[str, int]
    backbone_atoms_by_name: dict[str, int]


def _connector_contexts(
    mol: Chem.Mol,
    connector_params_by_key: Mapping[tuple[str, str], ConnectorParams],
    selector_instance_maps: Mapping[int, Any],
) -> dict[int, _ConnectorContext]:
    by_instance: dict[int, dict[str, object]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_manifest_source"):
            continue
        source = atom.GetProp("_poly_csp_manifest_source")
        if source != "connector":
            continue
        if not atom.HasProp("_poly_csp_selector_instance"):
            raise ValueError(f"Connector atom {atom.GetIdx()} is missing _poly_csp_selector_instance.")
        if not atom.HasProp("_poly_csp_selector_name"):
            raise ValueError(f"Connector atom {atom.GetIdx()} is missing _poly_csp_selector_name.")
        if not atom.HasProp("_poly_csp_site"):
            raise ValueError(f"Connector atom {atom.GetIdx()} is missing _poly_csp_site.")
        if not atom.HasProp("_poly_csp_residue_index"):
            raise ValueError(f"Connector atom {atom.GetIdx()} is missing _poly_csp_residue_index.")
        if not atom.HasProp("_poly_csp_atom_name"):
            raise ValueError(f"Connector atom {atom.GetIdx()} is missing _poly_csp_atom_name.")

        instance_id = int(atom.GetIntProp("_poly_csp_selector_instance"))
        selector_name = atom.GetProp("_poly_csp_selector_name")
        site = atom.GetProp("_poly_csp_site")
        residue_index = int(atom.GetIntProp("_poly_csp_residue_index"))
        key = (selector_name, site)
        if key not in connector_params_by_key:
            raise ValueError(f"No connector payload is available for selector/site {key!r}.")

        entry = by_instance.setdefault(
            instance_id,
            {
                "selector_name": selector_name,
                "site": site,
                "residue_index": residue_index,
                "connector_atoms_by_name": {},
            },
        )
        entry["connector_atoms_by_name"][atom.GetProp("_poly_csp_atom_name")] = int(atom.GetIdx())  # type: ignore[index]

    backbone_atoms = _backbone_name_maps(mol)
    out: dict[int, _ConnectorContext] = {}
    for instance_id, payload in by_instance.items():
        selector_name = str(payload["selector_name"])
        site = str(payload["site"])
        residue_index = int(payload["residue_index"])
        selector_map = selector_instance_maps.get(instance_id)
        if selector_map is None:
            raise ValueError(f"Missing selector-core mapping for connector instance {instance_id}.")
        connector_params = connector_params_by_key[(selector_name, site)]
        connector_atoms_by_name = dict(payload["connector_atoms_by_name"])  # type: ignore[arg-type]
        observed = set(connector_atoms_by_name)
        expected = set(connector_params.atom_params)
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extra = sorted(observed.difference(expected))
            raise ValueError(
                f"Connector instance atom-set mismatch for instance {instance_id}. "
                f"Missing={missing}, extra={extra}."
            )
        missing_role_atoms = {
            role_name: atom_name
            for role_name, atom_name in connector_params.connector_role_atom_names.items()
            if atom_name not in connector_atoms_by_name
        }
        if missing_role_atoms:
            raise ValueError(
                "Connector instance is missing connector-role atoms required by the "
                f"payload for instance {instance_id}: {missing_role_atoms!r}."
            )
        out[instance_id] = _ConnectorContext(
            instance_id=instance_id,
            selector_name=selector_name,
            site=site,
            residue_index=residue_index,
            selector_atoms_by_name=dict(selector_map.atom_index_by_name),
            connector_atoms_by_name=connector_atoms_by_name,
            backbone_atoms_by_name=dict(backbone_atoms.get(residue_index, {})),
        )
    return out


def _resolve_connector_token(context: _ConnectorContext, token: ConnectorToken) -> int:
    if token.source == "backbone":
        if token.atom_name not in context.backbone_atoms_by_name:
            raise ValueError(
                f"Backbone atom {token.atom_name!r} is missing from residue {context.residue_index}."
            )
        return context.backbone_atoms_by_name[token.atom_name]
    if token.source == "selector":
        if token.atom_name not in context.selector_atoms_by_name:
            raise ValueError(
                f"Selector atom {token.atom_name!r} is missing from instance {context.instance_id}."
            )
        return context.selector_atoms_by_name[token.atom_name]
    if token.atom_name not in context.connector_atoms_by_name:
        raise ValueError(
            f"Connector atom {token.atom_name!r} is missing from instance {context.instance_id}."
        )
    return context.connector_atoms_by_name[token.atom_name]


def _soft_sigma_nm(
    assigned_nonbonded: list[tuple[float, float, float] | None],
    atom_idx: int,
    atom: Chem.Atom,
) -> float:
    params = assigned_nonbonded[atom_idx]
    if params is not None:
        return float(params[1])
    z = atom.GetAtomicNum()
    if z <= 1:
        return 0.11
    if z == 6:
        return 0.17
    if z == 7:
        return 0.155
    if z == 8:
        return 0.152
    return 0.17


def create_system(
    mol: Chem.Mol,
    *,
    glycam_params: GlycamParams,
    selector_params_by_name: Mapping[str, SelectorFragmentParams] | None = None,
    connector_params_by_key: Mapping[tuple[str, str], ConnectorParams] | None = None,
    parameter_provenance: Mapping[str, object] | None = None,
    nonbonded_mode: str = "full",
    mixing_rules_cfg: Mapping[str, object] | None = None,
    repulsion_k_kj_per_mol_nm2: float = 800.0,
    repulsion_cutoff_nm: float = 0.6,
) -> SystemBuildResult:
    """Construct the canonical runtime system from real parameter sources."""
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Canonical runtime system construction requires a forcefield-domain molecule from build_forcefield_molecule()."
        )
    if nonbonded_mode not in {"soft", "full"}:
        raise ValueError(f"Unsupported nonbonded_mode {nonbonded_mode!r}.")

    selector_params_by_name = dict(selector_params_by_name or {})
    connector_params_by_key = dict(connector_params_by_key or {})
    positions_nm = _positions_nm_from_mol(mol)
    n_atoms = mol.GetNumAtoms()

    backbone_mapping = map_backbone_to_glycam(mol, glycam_params)
    selector_instance_maps = map_selector_instances(mol, selector_params_by_name)
    connector_context_by_instance = _connector_contexts(
        mol,
        connector_params_by_key,
        selector_instance_maps,
    )

    system = mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(_atomic_mass_dalton(atom.GetAtomicNum()) * unit.dalton)

    assigned_nonbonded: list[tuple[float, float, float] | None] = [None] * n_atoms
    source_manifest: dict[str, object] = {"glycam": dict(glycam_params.provenance)}

    atom_index_by_backbone_name = {
        (assignment.residue_index, assignment.glycam_atom_name): assignment.atom_index
        for assignment in backbone_mapping.assignments
    }
    residue_roles: list[str] = []
    if mol.HasProp("_poly_csp_dp"):
        dp = int(mol.GetIntProp("_poly_csp_dp"))
        for residue_index in range(dp):
            entries = [
                assignment
                for assignment in backbone_mapping.assignments
                if assignment.residue_index == residue_index
            ]
            if not entries:
                raise ValueError(f"Backbone residue {residue_index} is missing from the GLYCAM mapping.")
            residue_roles.append(entries[0].residue_role)
    missing_backbone_atom_keys: set[tuple[int, str]] = set()
    for residue_index, residue_role in enumerate(residue_roles):
        expected_names = set(glycam_params.residue_templates[residue_role].atom_names)
        observed_names = {
            assignment.glycam_atom_name
            for assignment in backbone_mapping.assignments
            if assignment.residue_index == residue_index
        }
        missing_backbone_atom_keys.update(
            (residue_index, atom_name)
            for atom_name in expected_names.difference(observed_names)
        )

    for assignment in backbone_mapping.assignments:
        params = glycam_params.atom_params[(assignment.residue_role, assignment.glycam_atom_name)]
        assigned_nonbonded[assignment.atom_index] = (
            float(params.charge_e),
            float(params.sigma_nm),
            float(params.epsilon_kj_per_mol),
        )

    for selector_name, params in selector_params_by_name.items():
        source_manifest.setdefault("selector", {})[selector_name] = {
            "source_prmtop": params.source_prmtop,
            "fragment_atom_count": params.fragment_atom_count,
        }
    for instance_id, mapping in selector_instance_maps.items():
        params = selector_params_by_name[mapping.selector_name]
        for atom_name, atom_idx in mapping.atom_index_by_name.items():
            atom_params = params.atom_params[atom_name]
            assigned_nonbonded[atom_idx] = (
                float(atom_params.charge_e),
                float(atom_params.sigma_nm),
                float(atom_params.epsilon_kj_per_mol),
            )

    for key, params in connector_params_by_key.items():
        source_manifest.setdefault("connector", {})[f"{key[0]}:{key[1]}"] = {
            "source_prmtop": params.source_prmtop,
            "fragment_atom_count": params.fragment_atom_count,
            "linkage_type": params.linkage_type,
            "connector_role_atom_names": dict(params.connector_role_atom_names),
        }
    for instance_id, context in connector_context_by_instance.items():
        params = connector_params_by_key[(context.selector_name, context.site)]
        validate_connector_params(params)
        for atom_name, atom_idx in context.connector_atoms_by_name.items():
            atom_params = params.atom_params[atom_name]
            assigned_nonbonded[atom_idx] = (
                float(atom_params.charge_e),
                float(atom_params.sigma_nm),
                float(atom_params.epsilon_kj_per_mol),
            )

    if any(params is None for params in assigned_nonbonded):
        missing = [idx for idx, params in enumerate(assigned_nonbonded) if params is None]
        raise ValueError(f"Canonical runtime system is missing atom parameters for indices {missing}.")

    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    torsion_force = mm.PeriodicTorsionForce()
    bond_owner_by_key: dict[tuple[int, int], str] = {}
    angle_owner_by_key: dict[tuple[int, int, int], str] = {}
    torsion_owner_by_key: dict[tuple[int, int, int, int], str] = {}

    def _resolve_backbone_tokens(anchor_residue: int, tokens) -> tuple[int, ...] | None:
        resolved: list[int] = []
        for token in tokens:
            key = (int(anchor_residue + token.residue_offset), token.atom_name)
            if key not in atom_index_by_backbone_name:
                if key in missing_backbone_atom_keys:
                    return None
                raise ValueError(
                    "Missing mapped GLYCAM atom while materializing system term: "
                    f"residue={key[0]}, atom={key[1]!r}."
                )
            resolved.append(atom_index_by_backbone_name[key])
        return tuple(resolved)

    for residue_index, residue_role in enumerate(residue_roles):
        residue_template = glycam_params.residue_templates[residue_role]
        for template in residue_template.bonds:
            resolved = _resolve_backbone_tokens(residue_index, template.atoms)
            if resolved is None:
                continue
            a, b = resolved
            _register_term_owner(
                bond_owner_by_key,
                _bond_key(a, b),
                f"backbone:{residue_role}",
                term_kind="bond",
            )
            _set_or_add_bond(bond_force, a, b, float(template.length_nm), float(template.k_kj_per_mol_nm2))
        for template in residue_template.angles:
            resolved = _resolve_backbone_tokens(residue_index, template.atoms)
            if resolved is None:
                continue
            a, b, c = resolved
            _register_term_owner(
                angle_owner_by_key,
                _angle_key(a, b, c),
                f"backbone:{residue_role}",
                term_kind="angle",
            )
            _set_or_add_angle(angle_force, a, b, c, float(template.theta0_rad), float(template.k_kj_per_mol_rad2))
        for template in residue_template.torsions:
            resolved = _resolve_backbone_tokens(residue_index, template.atoms)
            if resolved is None:
                continue
            a, b, c, d = resolved
            _register_term_owner(
                torsion_owner_by_key,
                _torsion_key(a, b, c, d),
                f"backbone:{residue_role}",
                term_kind="torsion",
            )
            torsion_force.addTorsion(a, b, c, d, int(template.periodicity), float(template.phase_rad), float(template.k_kj_per_mol))

    for left_residue in range(max(0, len(residue_roles) - 1)):
        pair = (residue_roles[left_residue], residue_roles[left_residue + 1])
        linkage_template = glycam_params.linkage_templates.get(pair)
        if linkage_template is None:
            raise ValueError(
                "No GLYCAM linkage template is available for residue-role pair "
                f"{pair[0]!r}->{pair[1]!r}."
            )
        for template in linkage_template.bonds:
            resolved = _resolve_backbone_tokens(left_residue, template.atoms)
            if resolved is None:
                continue
            a, b = resolved
            _register_term_owner(
                bond_owner_by_key,
                _bond_key(a, b),
                f"backbone_linkage:{pair[0]}->{pair[1]}",
                term_kind="bond",
            )
            _set_or_add_bond(bond_force, a, b, float(template.length_nm), float(template.k_kj_per_mol_nm2))
        for template in linkage_template.angles:
            resolved = _resolve_backbone_tokens(left_residue, template.atoms)
            if resolved is None:
                continue
            a, b, c = resolved
            _register_term_owner(
                angle_owner_by_key,
                _angle_key(a, b, c),
                f"backbone_linkage:{pair[0]}->{pair[1]}",
                term_kind="angle",
            )
            _set_or_add_angle(angle_force, a, b, c, float(template.theta0_rad), float(template.k_kj_per_mol_rad2))
        for template in linkage_template.torsions:
            resolved = _resolve_backbone_tokens(left_residue, template.atoms)
            if resolved is None:
                continue
            a, b, c, d = resolved
            _register_term_owner(
                torsion_owner_by_key,
                _torsion_key(a, b, c, d),
                f"backbone_linkage:{pair[0]}->{pair[1]}",
                term_kind="torsion",
            )
            torsion_force.addTorsion(a, b, c, d, int(template.periodicity), float(template.phase_rad), float(template.k_kj_per_mol))

    for instance_id, mapping in selector_instance_maps.items():
        params = selector_params_by_name[mapping.selector_name]
        for template in params.bonds:
            a = mapping.atom_index_by_name[template.atom_names[0]]
            b = mapping.atom_index_by_name[template.atom_names[1]]
            _register_term_owner(
                bond_owner_by_key,
                _bond_key(a, b),
                f"selector:{mapping.selector_name}:{instance_id}",
                term_kind="bond",
            )
            _set_or_add_bond(bond_force, a, b, float(template.length_nm), float(template.k_kj_per_mol_nm2))
        for template in params.angles:
            a = mapping.atom_index_by_name[template.atom_names[0]]
            b = mapping.atom_index_by_name[template.atom_names[1]]
            c = mapping.atom_index_by_name[template.atom_names[2]]
            _register_term_owner(
                angle_owner_by_key,
                _angle_key(a, b, c),
                f"selector:{mapping.selector_name}:{instance_id}",
                term_kind="angle",
            )
            _set_or_add_angle(angle_force, a, b, c, float(template.theta0_rad), float(template.k_kj_per_mol_rad2))
        for template in params.torsions:
            a = mapping.atom_index_by_name[template.atom_names[0]]
            b = mapping.atom_index_by_name[template.atom_names[1]]
            c = mapping.atom_index_by_name[template.atom_names[2]]
            d = mapping.atom_index_by_name[template.atom_names[3]]
            _register_term_owner(
                torsion_owner_by_key,
                _torsion_key(a, b, c, d),
                f"selector:{mapping.selector_name}:{instance_id}",
                term_kind="torsion",
            )
            torsion_force.addTorsion(a, b, c, d, int(template.periodicity), float(template.phase_rad), float(template.k_kj_per_mol))

    for instance_id, context in connector_context_by_instance.items():
        params = connector_params_by_key[(context.selector_name, context.site)]
        for template in params.bonds:
            a = _resolve_connector_token(context, template.atoms[0])
            b = _resolve_connector_token(context, template.atoms[1])
            _register_term_owner(
                bond_owner_by_key,
                _bond_key(a, b),
                f"connector:{context.selector_name}:{context.site}:{instance_id}",
                term_kind="bond",
            )
            _set_or_add_bond(bond_force, a, b, float(template.length_nm), float(template.k_kj_per_mol_nm2))
        for template in params.angles:
            a = _resolve_connector_token(context, template.atoms[0])
            b = _resolve_connector_token(context, template.atoms[1])
            c = _resolve_connector_token(context, template.atoms[2])
            _register_term_owner(
                angle_owner_by_key,
                _angle_key(a, b, c),
                f"connector:{context.selector_name}:{context.site}:{instance_id}",
                term_kind="angle",
            )
            _set_or_add_angle(angle_force, a, b, c, float(template.theta0_rad), float(template.k_kj_per_mol_rad2))
        for template in params.torsions:
            a = _resolve_connector_token(context, template.atoms[0])
            b = _resolve_connector_token(context, template.atoms[1])
            c = _resolve_connector_token(context, template.atoms[2])
            d = _resolve_connector_token(context, template.atoms[3])
            _register_term_owner(
                torsion_owner_by_key,
                _torsion_key(a, b, c, d),
                f"connector:{context.selector_name}:{context.site}:{instance_id}",
                term_kind="torsion",
            )
            torsion_force.addTorsion(a, b, c, d, int(template.periodicity), float(template.phase_rad), float(template.k_kj_per_mol))

    system.addForce(bond_force)
    system.addForce(angle_force)
    if torsion_force.getNumTorsions() > 0:
        system.addForce(torsion_force)

    bonds = [
        (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
    ]
    if nonbonded_mode == "full":
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        for charge_e, sigma_nm, epsilon_kj in assigned_nonbonded:
            nonbonded.addParticle(float(charge_e), float(sigma_nm), float(epsilon_kj))
        nonbonded.createExceptionsFromBonds(bonds, 1.0 / 1.2, 1.0 / 2.0)
        system.addForce(nonbonded)
        exception_summary = apply_mixing_rules(
            system=system,
            atom_map=build_atom_map(mol),
            mixing_rules_cfg=mixing_rules_cfg,
        )
        exception_summary["num_bonds"] = len(bonds)
        excluded = exclusion_pairs_from_mol(mol, exclude_13=True, exclude_14=False)
    else:
        repulsive = mm.CustomNonbondedForce(
            "k_rep*step(sigma-r)*(sigma-r)^2;"
            "sigma=0.5*(sigma1+sigma2)"
        )
        repulsive.addGlobalParameter("k_rep", float(repulsion_k_kj_per_mol_nm2))
        repulsive.addPerParticleParameter("sigma")
        repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        repulsive.setCutoffDistance(float(repulsion_cutoff_nm) * unit.nanometer)
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            repulsive.addParticle([_soft_sigma_nm(assigned_nonbonded, atom_idx, atom)])
        excluded = exclusion_pairs_from_mol(mol, exclude_13=True, exclude_14=False)
        for i, j in sorted(excluded):
            repulsive.addExclusion(int(i), int(j))
        system.addForce(repulsive)
        exception_summary = {
            "mode": "soft",
            "num_bonds": len(bonds),
            "num_exclusions": len(excluded),
        }

    return SystemBuildResult(
        system=system,
        positions_nm=positions_nm,
        excluded_pairs=excluded,
        nonbonded_mode=nonbonded_mode,
        topology_manifest=_topology_manifest(mol, backbone_mapping),
        component_counts=_component_counts(mol),
        exception_summary=exception_summary,
        source_manifest=_merge_source_manifest(source_manifest, parameter_provenance),
    )


def build_backbone_glycam_system(
    mol: Chem.Mol,
    glycam_params: GlycamParams,
) -> SystemBuildResult:
    """Build a pure-backbone system from the canonical runtime builder."""
    for atom in mol.GetAtoms():
        if atom.HasProp("_poly_csp_manifest_source") and atom.GetProp("_poly_csp_manifest_source") != "backbone":
            raise ValueError("build_backbone_glycam_system() supports pure backbone molecules only.")
    return create_system(
        mol,
        glycam_params=glycam_params,
        selector_params_by_name={},
        connector_params_by_key={},
        nonbonded_mode="full",
    )
