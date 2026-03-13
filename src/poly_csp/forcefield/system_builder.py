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
from poly_csp.forcefield.gaff import SelectorFragmentParams
from poly_csp.forcefield.glycam import GlycamParams
from poly_csp.forcefield.glycam_mapping import GlycamMappingResult, map_backbone_to_glycam
from poly_csp.forcefield.selector_mapping import map_selector_instances
from poly_csp.structure.pbc import get_box_vectors_A, get_box_vectors_nm


@dataclass(frozen=True)
class BondedTermSummary:
    bonds: int = 0
    angles: int = 0
    torsions: int = 0
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class ForceInventorySummary:
    forces: tuple[str, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemBuildResult:
    system: mm.System
    positions_nm: unit.Quantity
    excluded_pairs: set[tuple[int, int]]
    nonbonded_mode: str = "soft"
    topology_manifest: tuple[dict[str, object], ...] = ()
    component_counts: dict[str, int] = field(default_factory=dict)
    bonded_term_summary: BondedTermSummary = field(default_factory=BondedTermSummary)
    force_inventory: ForceInventorySummary = field(default_factory=ForceInventorySummary)
    exception_summary: dict[str, object] = field(default_factory=dict)
    source_manifest: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _ResolvedSystemInputs:
    component_counts: dict[str, int]
    positions_nm: unit.Quantity
    end_mode: str
    selector_params_by_name: dict[str, SelectorFragmentParams]
    connector_params_by_key: dict[tuple[str, str], ConnectorParams]
    backbone_mapping: GlycamMappingResult
    selector_instance_maps: dict[int, Any]
    connector_context_by_instance: dict[int, "_ConnectorContext"]
    assigned_nonbonded: tuple[tuple[float, float, float], ...]
    source_manifest: dict[str, object]
    atom_index_by_backbone_name: dict[tuple[int, str], int]
    residue_roles: tuple[str, ...]
    missing_backbone_atom_keys: frozenset[tuple[int, str]]


@dataclass
class _BondedAssembly:
    bond_force: mm.HarmonicBondForce = field(default_factory=mm.HarmonicBondForce)
    angle_force: mm.HarmonicAngleForce = field(default_factory=mm.HarmonicAngleForce)
    torsion_force: mm.PeriodicTorsionForce = field(default_factory=mm.PeriodicTorsionForce)
    bond_owner_by_key: dict[tuple[int, int], str] = field(default_factory=dict)
    angle_owner_by_key: dict[tuple[int, int, int], str] = field(default_factory=dict)
    torsion_owner_by_key: dict[tuple[int, int, int, int], str] = field(default_factory=dict)
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def _increment(self, owner: str, term_kind: str) -> None:
        source = _owner_summary_key(owner)
        bucket = self.by_source.setdefault(
            source,
            {"bonds": 0, "angles": 0, "torsions": 0},
        )
        bucket[term_kind] += 1

    def add_bond(self, a: int, b: int, r0, k, *, owner: str) -> None:
        key = _bond_key(a, b)
        if _register_term_owner(
            self.bond_owner_by_key,
            key,
            owner,
            term_kind="bond",
        ):
            self._increment(owner, "bonds")
        _set_or_add_bond(self.bond_force, a, b, r0, k)

    def add_angle(self, a: int, b: int, c: int, theta0, k, *, owner: str) -> None:
        key = _angle_key(a, b, c)
        if _register_term_owner(
            self.angle_owner_by_key,
            key,
            owner,
            term_kind="angle",
        ):
            self._increment(owner, "angles")
        _set_or_add_angle(self.angle_force, a, b, c, theta0, k)

    def add_torsion(
        self,
        a: int,
        b: int,
        c: int,
        d: int,
        periodicity: int,
        phase_rad: float,
        k_kj_per_mol: float,
        *,
        owner: str,
    ) -> None:
        key = _torsion_key(a, b, c, d)
        if _register_term_owner(
            self.torsion_owner_by_key,
            key,
            owner,
            term_kind="torsion",
        ):
            self._increment(owner, "torsions")
        self.torsion_force.addTorsion(a, b, c, d, periodicity, phase_rad, k_kj_per_mol)

    def summary(self) -> BondedTermSummary:
        return BondedTermSummary(
            bonds=int(self.bond_force.getNumBonds()),
            angles=int(self.angle_force.getNumAngles()),
            torsions=int(self.torsion_force.getNumTorsions()),
            by_source={
                source: {
                    "bonds": int(counts["bonds"]),
                    "angles": int(counts["angles"]),
                    "torsions": int(counts["torsions"]),
                }
                for source, counts in sorted(self.by_source.items())
            },
        )


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


def _owner_summary_key(owner: str) -> str:
    return str(owner).split(":", 1)[0]


def _register_term_owner(
    registry: dict[tuple[int, ...], str],
    key: tuple[int, ...],
    owner: str,
    *,
    term_kind: str,
) -> bool:
    existing = registry.get(key)
    if existing is None:
        registry[key] = owner
        return True
    if existing != owner:
        raise ValueError(
            f"Ambiguous {term_kind} ownership for atoms {key!r}: "
            f"{existing!r} vs {owner!r}."
        )
    return False


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


def _require_mol_prop(mol: Chem.Mol, name: str) -> str:
    if not mol.HasProp(name):
        raise ValueError(f"Forcefield-domain molecule is missing required property {name}.")
    return str(mol.GetProp(name))


def _validate_runtime_support_boundary(mol: Chem.Mol) -> None:
    polymer = _require_mol_prop(mol, "_poly_csp_polymer").strip().lower()
    representation = _require_mol_prop(mol, "_poly_csp_representation").strip().lower()
    end_mode = _require_mol_prop(mol, "_poly_csp_end_mode").strip().lower()

    if polymer not in {"amylose", "cellulose"}:
        raise ValueError(
            "Canonical runtime system currently supports only amylose/cellulose polymers; "
            f"got {polymer!r}."
        )
    if representation != "anhydro":
        raise ValueError(
            "Canonical runtime system currently supports only anhydro forcefield-domain "
            f"molecules; got {representation!r}."
        )
    if end_mode not in {"open", "periodic"}:
        raise ValueError(
            "Canonical runtime system currently supports only open and periodic "
            f"forcefield-domain molecules; got {end_mode!r}."
        )
    if end_mode == "periodic":
        if not mol.HasProp("_poly_csp_dp") or int(mol.GetIntProp("_poly_csp_dp")) < 2:
            raise ValueError("Periodic runtime system requires dp >= 2.")
        if get_box_vectors_nm(mol) is None:
            raise ValueError(
                "Periodic runtime system requires box vectors on the forcefield-domain molecule."
            )


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


def _resolve_system_inputs(
    mol: Chem.Mol,
    *,
    glycam_params: GlycamParams,
    selector_params_by_name: Mapping[str, SelectorFragmentParams] | None,
    connector_params_by_key: Mapping[tuple[str, str], ConnectorParams] | None,
) -> _ResolvedSystemInputs:
    _validate_runtime_support_boundary(mol)
    component_counts = _component_counts(mol)
    positions_nm = _positions_nm_from_mol(mol)
    end_mode = _require_mol_prop(mol, "_poly_csp_end_mode").strip().lower()

    resolved_selector_params = dict(selector_params_by_name or {})
    resolved_connector_params = dict(connector_params_by_key or {})
    backbone_mapping = map_backbone_to_glycam(mol, glycam_params)
    selector_instance_maps = map_selector_instances(mol, resolved_selector_params)
    connector_context_by_instance = _connector_contexts(
        mol,
        resolved_connector_params,
        selector_instance_maps,
    )

    assigned_nonbonded: list[tuple[float, float, float] | None] = [None] * mol.GetNumAtoms()
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
                raise ValueError(
                    f"Backbone residue {residue_index} is missing from the GLYCAM mapping."
                )
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

    for selector_name, params in resolved_selector_params.items():
        source_manifest.setdefault("selector", {})[selector_name] = {
            "source_prmtop": params.source_prmtop,
            "fragment_atom_count": params.fragment_atom_count,
        }
    for mapping in selector_instance_maps.values():
        params = resolved_selector_params[mapping.selector_name]
        for atom_name, atom_idx in mapping.atom_index_by_name.items():
            atom_params = params.atom_params[atom_name]
            assigned_nonbonded[atom_idx] = (
                float(atom_params.charge_e),
                float(atom_params.sigma_nm),
                float(atom_params.epsilon_kj_per_mol),
            )

    for key, params in resolved_connector_params.items():
        source_manifest.setdefault("connector", {})[f"{key[0]}:{key[1]}"] = {
            "source_prmtop": params.source_prmtop,
            "fragment_atom_count": params.fragment_atom_count,
            "linkage_type": params.linkage_type,
            "connector_role_atom_names": dict(params.connector_role_atom_names),
        }
    for context in connector_context_by_instance.values():
        params = resolved_connector_params[(context.selector_name, context.site)]
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

    return _ResolvedSystemInputs(
        component_counts=component_counts,
        positions_nm=positions_nm,
        end_mode=end_mode,
        selector_params_by_name=resolved_selector_params,
        connector_params_by_key=resolved_connector_params,
        backbone_mapping=backbone_mapping,
        selector_instance_maps=selector_instance_maps,
        connector_context_by_instance=connector_context_by_instance,
        assigned_nonbonded=tuple(
            (float(item[0]), float(item[1]), float(item[2]))
            for item in assigned_nonbonded
            if item is not None
        ),
        source_manifest=source_manifest,
        atom_index_by_backbone_name=atom_index_by_backbone_name,
        residue_roles=tuple(residue_roles),
        missing_backbone_atom_keys=frozenset(missing_backbone_atom_keys),
    )


def _resolve_backbone_tokens(
    inputs: _ResolvedSystemInputs,
    anchor_residue: int,
    tokens,
) -> tuple[int, ...] | None:
    resolved: list[int] = []
    residue_count = len(inputs.residue_roles)
    for token in tokens:
        residue_index = int(anchor_residue + token.residue_offset)
        if inputs.end_mode == "periodic":
            residue_index %= residue_count
        key = (residue_index, token.atom_name)
        if key not in inputs.atom_index_by_backbone_name:
            if key in inputs.missing_backbone_atom_keys:
                return None
            raise ValueError(
                "Missing mapped GLYCAM atom while materializing system term: "
                f"residue={key[0]}, atom={key[1]!r}."
            )
        resolved.append(inputs.atom_index_by_backbone_name[key])
    return tuple(resolved)


def _materialize_backbone_terms(
    assembly: _BondedAssembly,
    inputs: _ResolvedSystemInputs,
    glycam_params: GlycamParams,
) -> None:
    for residue_index, residue_role in enumerate(inputs.residue_roles):
        residue_template = glycam_params.residue_templates[residue_role]
        for template in residue_template.bonds:
            resolved = _resolve_backbone_tokens(inputs, residue_index, template.atoms)
            if resolved is None:
                continue
            a, b = resolved
            assembly.add_bond(
                a,
                b,
                float(template.length_nm),
                float(template.k_kj_per_mol_nm2),
                owner=f"backbone:{residue_role}",
            )
        for template in residue_template.angles:
            resolved = _resolve_backbone_tokens(inputs, residue_index, template.atoms)
            if resolved is None:
                continue
            a, b, c = resolved
            assembly.add_angle(
                a,
                b,
                c,
                float(template.theta0_rad),
                float(template.k_kj_per_mol_rad2),
                owner=f"backbone:{residue_role}",
            )
        for template in residue_template.torsions:
            resolved = _resolve_backbone_tokens(inputs, residue_index, template.atoms)
            if resolved is None:
                continue
            a, b, c, d = resolved
            assembly.add_torsion(
                a,
                b,
                c,
                d,
                int(template.periodicity),
                float(template.phase_rad),
                float(template.k_kj_per_mol),
                owner=f"backbone:{residue_role}",
            )

    residue_count = len(inputs.residue_roles)
    if inputs.end_mode == "periodic":
        linkage_indices = range(residue_count)
    else:
        linkage_indices = range(max(0, residue_count - 1))

    for left_residue in linkage_indices:
        right_residue = (left_residue + 1) % residue_count
        pair = (inputs.residue_roles[left_residue], inputs.residue_roles[right_residue])
        linkage_template = glycam_params.linkage_templates.get(pair)
        if linkage_template is None:
            raise ValueError(
                "No GLYCAM linkage template is available for residue-role pair "
                f"{pair[0]!r}->{pair[1]!r}."
            )
        for template in linkage_template.bonds:
            resolved = _resolve_backbone_tokens(inputs, left_residue, template.atoms)
            if resolved is None:
                continue
            a, b = resolved
            assembly.add_bond(
                a,
                b,
                float(template.length_nm),
                float(template.k_kj_per_mol_nm2),
                owner=f"backbone_linkage:{pair[0]}->{pair[1]}",
            )
        for template in linkage_template.angles:
            resolved = _resolve_backbone_tokens(inputs, left_residue, template.atoms)
            if resolved is None:
                continue
            a, b, c = resolved
            assembly.add_angle(
                a,
                b,
                c,
                float(template.theta0_rad),
                float(template.k_kj_per_mol_rad2),
                owner=f"backbone_linkage:{pair[0]}->{pair[1]}",
            )
        for template in linkage_template.torsions:
            resolved = _resolve_backbone_tokens(inputs, left_residue, template.atoms)
            if resolved is None:
                continue
            a, b, c, d = resolved
            assembly.add_torsion(
                a,
                b,
                c,
                d,
                int(template.periodicity),
                float(template.phase_rad),
                float(template.k_kj_per_mol),
                owner=f"backbone_linkage:{pair[0]}->{pair[1]}",
            )


def _materialize_selector_terms(
    assembly: _BondedAssembly,
    inputs: _ResolvedSystemInputs,
) -> None:
    for instance_id, mapping in inputs.selector_instance_maps.items():
        params = inputs.selector_params_by_name[mapping.selector_name]
        owner = f"selector:{mapping.selector_name}:{instance_id}"
        for template in params.bonds:
            assembly.add_bond(
                mapping.atom_index_by_name[template.atom_names[0]],
                mapping.atom_index_by_name[template.atom_names[1]],
                float(template.length_nm),
                float(template.k_kj_per_mol_nm2),
                owner=owner,
            )
        for template in params.angles:
            assembly.add_angle(
                mapping.atom_index_by_name[template.atom_names[0]],
                mapping.atom_index_by_name[template.atom_names[1]],
                mapping.atom_index_by_name[template.atom_names[2]],
                float(template.theta0_rad),
                float(template.k_kj_per_mol_rad2),
                owner=owner,
            )
        for template in params.torsions:
            assembly.add_torsion(
                mapping.atom_index_by_name[template.atom_names[0]],
                mapping.atom_index_by_name[template.atom_names[1]],
                mapping.atom_index_by_name[template.atom_names[2]],
                mapping.atom_index_by_name[template.atom_names[3]],
                int(template.periodicity),
                float(template.phase_rad),
                float(template.k_kj_per_mol),
                owner=owner,
            )


def _materialize_connector_terms(
    assembly: _BondedAssembly,
    inputs: _ResolvedSystemInputs,
) -> None:
    for instance_id, context in inputs.connector_context_by_instance.items():
        params = inputs.connector_params_by_key[(context.selector_name, context.site)]
        owner = f"connector:{context.selector_name}:{context.site}:{instance_id}"
        for template in params.bonds:
            assembly.add_bond(
                _resolve_connector_token(context, template.atoms[0]),
                _resolve_connector_token(context, template.atoms[1]),
                float(template.length_nm),
                float(template.k_kj_per_mol_nm2),
                owner=owner,
            )
        for template in params.angles:
            assembly.add_angle(
                _resolve_connector_token(context, template.atoms[0]),
                _resolve_connector_token(context, template.atoms[1]),
                _resolve_connector_token(context, template.atoms[2]),
                float(template.theta0_rad),
                float(template.k_kj_per_mol_rad2),
                owner=owner,
            )
        for template in params.torsions:
            assembly.add_torsion(
                _resolve_connector_token(context, template.atoms[0]),
                _resolve_connector_token(context, template.atoms[1]),
                _resolve_connector_token(context, template.atoms[2]),
                _resolve_connector_token(context, template.atoms[3]),
                int(template.periodicity),
                float(template.phase_rad),
                float(template.k_kj_per_mol),
                owner=owner,
            )


def _materialize_bonded_terms(
    inputs: _ResolvedSystemInputs,
    glycam_params: GlycamParams,
) -> _BondedAssembly:
    assembly = _BondedAssembly()
    _materialize_backbone_terms(assembly, inputs, glycam_params)
    _materialize_selector_terms(assembly, inputs)
    _materialize_connector_terms(assembly, inputs)
    return assembly


def _add_nonbonded_force(
    system: mm.System,
    mol: Chem.Mol,
    *,
    assigned_nonbonded: Sequence[tuple[float, float, float]],
    nonbonded_mode: str,
    periodic: bool,
    mixing_rules_cfg: Mapping[str, object] | None,
    repulsion_k_kj_per_mol_nm2: float,
    repulsion_cutoff_nm: float,
    soft_exclude_14: bool,
    anti_stacking_sigma_scale: float,
) -> tuple[set[tuple[int, int]], dict[str, object]]:
    bonds = [
        (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
    ]
    excluded = exclusion_pairs_from_mol(
        mol,
        exclude_13=True,
        exclude_14=(bool(soft_exclude_14) if nonbonded_mode == "soft" else False),
    )

    if nonbonded_mode == "full":
        nonbonded = mm.NonbondedForce()
        if periodic:
            cutoff_nm = _periodic_cutoff_nm(mol)
            nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
            nonbonded.setCutoffDistance(float(cutoff_nm) * unit.nanometer)
        else:
            cutoff_nm = None
            nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        for charge_e, sigma_nm, epsilon_kj in assigned_nonbonded:
            nonbonded.addParticle(float(charge_e), float(sigma_nm), float(epsilon_kj))
        nonbonded.createExceptionsFromBonds(bonds, 1.0, 1.0)
        for exception_idx in range(nonbonded.getNumExceptions()):
            atom_a, atom_b, charge_prod, sigma, epsilon = nonbonded.getExceptionParameters(
                exception_idx
            )
            pair = _bond_key(int(atom_a), int(atom_b))
            if pair not in excluded:
                continue
            nonbonded.setExceptionParameters(
                exception_idx,
                atom_a,
                atom_b,
                charge_prod * 0.0,
                sigma,
                epsilon * 0.0,
            )
        system.addForce(nonbonded)
        exception_summary = apply_mixing_rules(
            nonbonded=nonbonded,
            mol=mol,
            mixing_rules_cfg=mixing_rules_cfg,
        )
        exception_summary.update(
            {
                "mode": "full",
                "force_kind": "NonbondedForce",
                "num_bonds": len(bonds),
                "num_exclusions": len(excluded),
                "num_particles": len(assigned_nonbonded),
                "periodic": bool(periodic),
                "cutoff_nm": (None if cutoff_nm is None else float(cutoff_nm)),
            }
        )
        return excluded, exception_summary

    repulsive = mm.CustomNonbondedForce(
        "k_rep*step(sigma-r)*(sigma-r)^2;"
        "sigma=0.5*(sigma1+sigma2)"
    )
    repulsive.addGlobalParameter("k_rep", float(repulsion_k_kj_per_mol_nm2))
    repulsive.addPerParticleParameter("sigma")
    if periodic:
        cutoff_nm = _periodic_cutoff_nm(mol)
        repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        repulsive.setCutoffDistance(float(cutoff_nm) * unit.nanometer)
    else:
        cutoff_nm = float(repulsion_cutoff_nm)
        repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        repulsive.setCutoffDistance(float(cutoff_nm) * unit.nanometer)
    scaled_aromatic_selector_atoms = 0
    for atom_idx in range(mol.GetNumAtoms()):
        sigma_nm, was_scaled = _soft_sigma_nm(
            assigned_nonbonded,
            mol,
            atom_idx,
            anti_stacking_sigma_scale=float(anti_stacking_sigma_scale),
        )
        repulsive.addParticle([sigma_nm])
        if was_scaled:
            scaled_aromatic_selector_atoms += 1
    for i, j in sorted(excluded):
        repulsive.addExclusion(int(i), int(j))
    system.addForce(repulsive)
    return excluded, {
        "mode": "soft",
        "force_kind": "CustomNonbondedForce",
        "num_bonds": len(bonds),
        "num_exclusions": len(excluded),
        "num_particles": len(assigned_nonbonded),
        "periodic": bool(periodic),
        "repulsion_k_kj_per_mol_nm2": float(repulsion_k_kj_per_mol_nm2),
        "repulsion_cutoff_nm": float(cutoff_nm),
        "soft_exclude_14": bool(soft_exclude_14),
        "anti_stacking_sigma_scale": float(anti_stacking_sigma_scale),
        "scaled_aromatic_selector_atoms": int(scaled_aromatic_selector_atoms),
    }


def _collect_force_inventory(system: mm.System) -> ForceInventorySummary:
    forces = tuple(system.getForce(i).__class__.__name__ for i in range(system.getNumForces()))
    counts: dict[str, int] = {}
    for force_name in forces:
        counts[force_name] = counts.get(force_name, 0) + 1
    return ForceInventorySummary(
        forces=forces,
        counts={name: int(count) for name, count in sorted(counts.items())},
    )


def _soft_sigma_nm(
    assigned_nonbonded: Sequence[tuple[float, float, float]],
    mol: Chem.Mol,
    atom_idx: int,
    *,
    anti_stacking_sigma_scale: float,
) -> tuple[float, bool]:
    params = assigned_nonbonded[atom_idx]
    sigma_nm = float(params[1])
    atom = mol.GetAtomWithIdx(int(atom_idx))
    if (
        atom.GetAtomicNum() > 1
        and atom.GetIsAromatic()
        and atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "selector"
    ):
        anti_stacking_sigma_scale = float(anti_stacking_sigma_scale)
        if anti_stacking_sigma_scale != 1.0:
            return sigma_nm * anti_stacking_sigma_scale, True
    return sigma_nm, False


def _periodic_cutoff_nm(mol: Chem.Mol) -> float:
    box_vectors_A = get_box_vectors_A(mol)
    if box_vectors_A is None:
        raise ValueError(
            "Periodic runtime system requires box vectors on the forcefield-domain molecule."
        )
    min_length_A = min(float(length) for length in box_vectors_A)
    return float(0.49 * (min_length_A / 10.0))


def _configure_periodic_system(
    system: mm.System,
    mol: Chem.Mol,
    bonded_assembly: _BondedAssembly,
) -> None:
    box_vectors_nm = get_box_vectors_nm(mol)
    if box_vectors_nm is None:
        raise ValueError(
            "Periodic runtime system requires box vectors on the forcefield-domain molecule."
        )
    system.setDefaultPeriodicBoxVectors(*box_vectors_nm)
    bonded_assembly.bond_force.setUsesPeriodicBoundaryConditions(True)
    bonded_assembly.angle_force.setUsesPeriodicBoundaryConditions(True)
    bonded_assembly.torsion_force.setUsesPeriodicBoundaryConditions(True)


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
    soft_exclude_14: bool = False,
    anti_stacking_sigma_scale: float = 1.0,
) -> SystemBuildResult:
    """Construct the canonical runtime system from real parameter sources."""
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Canonical runtime system construction requires a forcefield-domain molecule from build_forcefield_molecule()."
        )
    if nonbonded_mode not in {"soft", "full"}:
        raise ValueError(f"Unsupported nonbonded_mode {nonbonded_mode!r}.")

    inputs = _resolve_system_inputs(
        mol,
        glycam_params=glycam_params,
        selector_params_by_name=selector_params_by_name,
        connector_params_by_key=connector_params_by_key,
    )
    periodic = bool(inputs.end_mode == "periodic")

    system = mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(_atomic_mass_dalton(atom.GetAtomicNum()) * unit.dalton)

    bonded_assembly = _materialize_bonded_terms(inputs, glycam_params)
    if periodic:
        _configure_periodic_system(system, mol, bonded_assembly)
    system.addForce(bonded_assembly.bond_force)
    system.addForce(bonded_assembly.angle_force)
    if bonded_assembly.torsion_force.getNumTorsions() > 0:
        system.addForce(bonded_assembly.torsion_force)

    excluded, exception_summary = _add_nonbonded_force(
        system,
        mol,
        assigned_nonbonded=inputs.assigned_nonbonded,
        nonbonded_mode=nonbonded_mode,
        periodic=periodic,
        mixing_rules_cfg=mixing_rules_cfg,
        repulsion_k_kj_per_mol_nm2=float(repulsion_k_kj_per_mol_nm2),
        repulsion_cutoff_nm=float(repulsion_cutoff_nm),
        soft_exclude_14=bool(soft_exclude_14),
        anti_stacking_sigma_scale=float(anti_stacking_sigma_scale),
    )
    force_inventory = _collect_force_inventory(system)

    return SystemBuildResult(
        system=system,
        positions_nm=inputs.positions_nm,
        excluded_pairs=excluded,
        nonbonded_mode=nonbonded_mode,
        topology_manifest=_topology_manifest(mol, inputs.backbone_mapping),
        component_counts=inputs.component_counts,
        bonded_term_summary=bonded_assembly.summary(),
        force_inventory=force_inventory,
        exception_summary=exception_summary,
        source_manifest=_merge_source_manifest(inputs.source_manifest, parameter_provenance),
    )


def build_backbone_glycam_system(
    mol: Chem.Mol,
    glycam_params: GlycamParams,
) -> SystemBuildResult:
    """Build a pure-backbone specialization of the canonical runtime builder."""
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
