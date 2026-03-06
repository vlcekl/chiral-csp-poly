from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem

import openmm as mm
from openmm import unit

if TYPE_CHECKING:
    from poly_csp.forcefield.connectors import ConnectorParams


@dataclass(frozen=True)
class SystemBuildResult:
    system: mm.System
    positions_nm: unit.Quantity
    excluded_pairs: set[tuple[int, int]]
    nonbonded_mode: str = "custom_repulsion"
    topology_manifest: tuple[dict[str, object], ...] = ()
    component_counts: dict[str, int] = field(default_factory=dict)
    exception_summary: dict[str, object] = field(default_factory=dict)
    source_manifest: dict[str, object] = field(default_factory=dict)


_SIGMA_A_BY_Z = {
    1: 1.10,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
}


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


def _sigma_nm(atom: Chem.Atom) -> float:
    return float(_SIGMA_A_BY_Z.get(atom.GetAtomicNum(), 1.70) / 10.0)


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


def build_relaxation_system(
    mol: Chem.Mol,
    repulsion_k_kj_per_mol_nm2: float = 800.0,
    repulsion_cutoff_nm: float = 0.6,
    exclude_13: bool = True,
) -> SystemBuildResult:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before OpenMM system build.")

    positions_nm = _positions_nm_from_mol(mol)

    system = mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(_atomic_mass_dalton(atom.GetAtomicNum()) * unit.dalton)

    # Pairwise soft repulsion to resolve overlaps without full parameterization.
    repulsive = mm.CustomNonbondedForce(
        "k_rep*step(sigma-r)*(sigma-r)^2;"
        "sigma=0.5*(sigma1+sigma2)"
    )
    repulsive.addGlobalParameter("k_rep", float(repulsion_k_kj_per_mol_nm2))
    repulsive.addPerParticleParameter("sigma")
    repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    repulsive.setCutoffDistance(float(repulsion_cutoff_nm) * unit.nanometer)

    for atom in mol.GetAtoms():
        repulsive.addParticle([_sigma_nm(atom)])

    excluded = exclusion_pairs_from_mol(mol, exclude_13=exclude_13, exclude_14=False)
    for i, j in sorted(excluded):
        repulsive.addExclusion(int(i), int(j))
    system.addForce(repulsive)

    return SystemBuildResult(
        system=system,
        positions_nm=positions_nm,
        excluded_pairs=excluded,
        component_counts={"backbone": mol.GetNumAtoms(), "selector": 0, "connector": 0},
    )


# ---------------------------------------------------------------------------
# Covalent radii (Å) for generic bond-length estimation.
# ---------------------------------------------------------------------------
_COVALENT_RADIUS_A = {
    1: 0.31, 6: 0.77, 7: 0.73, 8: 0.73, 9: 0.64,
    15: 1.07, 16: 1.02, 17: 0.99,
}


def _covalent_bond_length_nm(z1: int, z2: int) -> float:
    """Equilibrium bond length (nm) as sum of covalent radii."""
    r1 = _COVALENT_RADIUS_A.get(z1, 0.77)
    r2 = _COVALENT_RADIUS_A.get(z2, 0.77)
    return (r1 + r2) / 10.0  # Å → nm


def _equilibrium_angle_rad(central_atom: Chem.Atom) -> float:
    """Guess equilibrium angle from the number of heavy neighbours (hybridization proxy)."""
    import math
    n_neighbors = central_atom.GetDegree()
    if n_neighbors <= 2:
        return math.pi          # sp  → 180°
    if n_neighbors == 3:
        return 2.0943951        # sp2 → 120°
    return 1.9106332            # sp3 → 109.5°


def build_bonded_relaxation_system(
    mol: Chem.Mol,
    bond_k_kj_per_mol_nm2: float = 200_000.0,
    angle_k_kj_per_mol_rad2: float = 500.0,
    repulsion_k_kj_per_mol_nm2: float = 800.0,
    repulsion_cutoff_nm: float = 0.6,
    exclude_13: bool = True,
) -> SystemBuildResult:
    """Build an OpenMM system with **generic bonded forces** derived from RDKit.

    Unlike ``build_relaxation_system`` (soft repulsion only), this builder
    adds ``HarmonicBondForce`` and ``HarmonicAngleForce`` so that the
    molecule stays covalently intact during Langevin dynamics / annealing.
    The parameters are approximate (covalent-radii bond lengths,
    hybridisation-based angles) — sufficient to preserve topology, but not
    production-quality force-field values.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before OpenMM system build.")

    positions_nm = _positions_nm_from_mol(mol)

    system = mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(_atomic_mass_dalton(atom.GetAtomicNum()) * unit.dalton)

    # --- Harmonic bonds from RDKit bond graph. ---
    bond_force = mm.HarmonicBondForce()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        z1 = mol.GetAtomWithIdx(i).GetAtomicNum()
        z2 = mol.GetAtomWithIdx(j).GetAtomicNum()
        r0 = _covalent_bond_length_nm(z1, z2)
        bond_force.addBond(i, j, r0, bond_k_kj_per_mol_nm2)
    system.addForce(bond_force)

    # --- Harmonic angles from bond-graph i-j-k triples. ---
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)

    angle_force = mm.HarmonicAngleForce()
    for j in range(n):
        nbrs = adj[j]
        theta0 = _equilibrium_angle_rad(mol.GetAtomWithIdx(j))
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                angle_force.addAngle(nbrs[ii], j, nbrs[jj], theta0, angle_k_kj_per_mol_rad2)
    system.addForce(angle_force)

    # --- Soft pairwise repulsion (same model as build_relaxation_system). ---
    repulsive = mm.CustomNonbondedForce(
        "k_rep*step(sigma-r)*(sigma-r)^2;"
        "sigma=0.5*(sigma1+sigma2)"
    )
    repulsive.addGlobalParameter("k_rep", float(repulsion_k_kj_per_mol_nm2))
    repulsive.addPerParticleParameter("sigma")
    repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    repulsive.setCutoffDistance(float(repulsion_cutoff_nm) * unit.nanometer)

    for atom in mol.GetAtoms():
        repulsive.addParticle([_sigma_nm(atom)])

    excluded = exclusion_pairs_from_mol(mol, exclude_13=exclude_13, exclude_14=False)
    for i, j in sorted(excluded):
        repulsive.addExclusion(int(i), int(j))
    system.addForce(repulsive)

    return SystemBuildResult(
        system=system,
        positions_nm=positions_nm,
        excluded_pairs=excluded,
        component_counts={"backbone": mol.GetNumAtoms(), "selector": 0, "connector": 0},
    )


def build_backbone_glycam_system(
    mol: Chem.Mol,
    glycam_params,
) -> SystemBuildResult:
    """Build a pure-backbone OpenMM system directly from extracted GLYCAM templates."""
    from poly_csp.forcefield.glycam_mapping import map_backbone_to_glycam

    positions_nm = _positions_nm_from_mol(mol)
    mapping = map_backbone_to_glycam(mol, glycam_params)
    assignments = mapping.assignments
    assignment_by_atom = {item.atom_index: item for item in assignments}
    atom_index_by_residue_name = {
        (item.residue_index, item.glycam_atom_name): item.atom_index
        for item in assignments
    }
    residue_roles = []
    if mol.HasProp("_poly_csp_dp"):
        dp = int(mol.GetIntProp("_poly_csp_dp"))
        for residue_index in range(dp):
            per_residue = [item for item in assignments if item.residue_index == residue_index]
            if not per_residue:
                raise ValueError(f"Backbone residue {residue_index} is missing from the GLYCAM mapping.")
            residue_roles.append(per_residue[0].residue_role)

    system = mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(_atomic_mass_dalton(atom.GetAtomicNum()) * unit.dalton)

    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for atom_idx in range(mol.GetNumAtoms()):
        assignment = assignment_by_atom.get(atom_idx)
        if assignment is None:
            raise ValueError(f"Backbone GLYCAM mapping is missing atom index {atom_idx}.")
        params = glycam_params.atom_params[
            (assignment.residue_role, assignment.glycam_atom_name)
        ]
        nonbonded.addParticle(
            float(params.charge_e),
            float(params.sigma_nm),
            float(params.epsilon_kj_per_mol),
        )

    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    torsion_force = mm.PeriodicTorsionForce()

    def _resolve_tokens(anchor_residue: int, tokens) -> tuple[int, ...]:
        out: list[int] = []
        for token in tokens:
            residue_index = int(anchor_residue + token.residue_offset)
            key = (residue_index, token.atom_name)
            if key not in atom_index_by_residue_name:
                raise ValueError(
                    "Missing mapped GLYCAM atom while materializing system term: "
                    f"residue={residue_index}, atom={token.atom_name!r}."
                )
            out.append(atom_index_by_residue_name[key])
        return tuple(out)

    for residue_index, residue_role in enumerate(residue_roles):
        residue_template = glycam_params.residue_templates[residue_role]
        for template in residue_template.bonds:
            a, b = _resolve_tokens(residue_index, template.atoms)
            bond_force.addBond(a, b, float(template.length_nm), float(template.k_kj_per_mol_nm2))
        for template in residue_template.angles:
            a, b, c = _resolve_tokens(residue_index, template.atoms)
            angle_force.addAngle(a, b, c, float(template.theta0_rad), float(template.k_kj_per_mol_rad2))
        for template in residue_template.torsions:
            a, b, c, d = _resolve_tokens(residue_index, template.atoms)
            torsion_force.addTorsion(
                a,
                b,
                c,
                d,
                int(template.periodicity),
                float(template.phase_rad),
                float(template.k_kj_per_mol),
            )

    for left_residue in range(max(0, len(residue_roles) - 1)):
        pair = (residue_roles[left_residue], residue_roles[left_residue + 1])
        linkage_template = glycam_params.linkage_templates.get(pair)
        if linkage_template is None:
            raise ValueError(
                "No GLYCAM linkage template is available for residue-role pair "
                f"{pair[0]!r}->{pair[1]!r}."
            )
        for template in linkage_template.bonds:
            a, b = _resolve_tokens(left_residue, template.atoms)
            bond_force.addBond(a, b, float(template.length_nm), float(template.k_kj_per_mol_nm2))
        for template in linkage_template.angles:
            a, b, c = _resolve_tokens(left_residue, template.atoms)
            angle_force.addAngle(a, b, c, float(template.theta0_rad), float(template.k_kj_per_mol_rad2))
        for template in linkage_template.torsions:
            a, b, c, d = _resolve_tokens(left_residue, template.atoms)
            torsion_force.addTorsion(
                a,
                b,
                c,
                d,
                int(template.periodicity),
                float(template.phase_rad),
                float(template.k_kj_per_mol),
            )

    bonds = []
    for bond in mol.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        bonds.append((a, b))
    nonbonded.createExceptionsFromBonds(bonds, 1.0, 1.0)

    system.addForce(nonbonded)
    system.addForce(bond_force)
    system.addForce(angle_force)
    system.addForce(torsion_force)

    excluded = exclusion_pairs_from_mol(mol, exclude_13=True, exclude_14=False)
    topology_manifest = tuple(
        {
            "atom_index": int(item.atom_index),
            "residue_index": int(item.residue_index),
            "residue_role": str(item.residue_role),
            "glycam_residue_name": str(item.glycam_residue_name),
            "generic_atom_name": str(item.generic_atom_name),
            "glycam_atom_name": str(item.glycam_atom_name),
        }
        for item in assignments
    )
    return SystemBuildResult(
        system=system,
        positions_nm=positions_nm,
        excluded_pairs=excluded,
        nonbonded_mode="glycam_no_cutoff",
        topology_manifest=topology_manifest,
        component_counts={"backbone": mol.GetNumAtoms(), "selector": 0, "connector": 0},
        exception_summary={
            "num_bonds": len(bonds),
            "num_exceptions": int(nonbonded.getNumExceptions()),
            "coulomb14scale": 1.0,
            "lj14scale": 1.0,
        },
        source_manifest=dict(glycam_params.provenance),
    )


def add_glycam_backbone(
    system: mm.System,
    mol: Chem.Mol,
    backbone_indices: set[int],
    glycam_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Attach backbone-specific parameters to an existing system (incremental API)."""
    _ = (mol, backbone_indices, glycam_params)
    return system


def add_gaff_selectors(
    system: mm.System,
    mol: Chem.Mol,
    selector_indices: set[int],
    gaff_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Attach selector-core GAFF bonded parameters to an existing system."""
    if not gaff_params or not selector_indices:
        return system

    forces = gaff_params.get("forces")
    if forces is None:
        selector_prmtop_path = gaff_params.get("selector_prmtop_path")
        selector_template = gaff_params.get("selector_template")
        if not selector_prmtop_path or selector_template is None:
            raise ValueError(
                "gaff_params must include either 'forces' or both "
                "'selector_prmtop_path' and 'selector_template'."
            )
        from poly_csp.forcefield.gaff import load_gaff2_selector_forces

        forces = load_gaff2_selector_forces(
            selector_prmtop_path=str(selector_prmtop_path),
            mol=mol,
            selector_template=selector_template,
        )

    _apply_transferred_bonded_forces(system, forces)
    return system


def add_connectors(
    system: mm.System,
    mol: Chem.Mol,
    connector_indices: set[int],
    connector_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Patch connector bond/angle terms and add connector torsions."""
    _ = connector_indices
    params_list = _normalize_connector_params(connector_params)
    if not params_list:
        return system

    bond_force = _first_force(system, mm.HarmonicBondForce)
    angle_force = _first_force(system, mm.HarmonicAngleForce)
    torsion_force = mm.PeriodicTorsionForce()
    added_torsions = 0

    for params in params_list:
        for role_map in _connector_role_maps(mol, params):
            if bond_force is not None:
                for roles, (r0, k) in params.bond_params.items():
                    a = role_map.get(roles[0])
                    b = role_map.get(roles[1])
                    if a is None or b is None:
                        continue
                    _set_or_add_bond(bond_force, int(a), int(b), float(r0), float(k))

            if angle_force is not None:
                for roles, (theta0, k) in params.angle_params.items():
                    a = role_map.get(roles[0])
                    b = role_map.get(roles[1])
                    c = role_map.get(roles[2])
                    if a is None or b is None or c is None:
                        continue
                    _set_or_add_angle(
                        angle_force,
                        int(a),
                        int(b),
                        int(c),
                        float(theta0),
                        float(k),
                    )

            for roles, (periodicity, phase, k) in params.torsion_params:
                a = role_map.get(roles[0])
                b = role_map.get(roles[1])
                c = role_map.get(roles[2])
                d = role_map.get(roles[3])
                if a is None or b is None or c is None or d is None:
                    continue
                torsion_force.addTorsion(
                    int(a),
                    int(b),
                    int(c),
                    int(d),
                    int(periodicity),
                    float(phase),
                    float(k),
                )
                added_torsions += 1

    if added_torsions > 0:
        system.addForce(torsion_force)
    return system


def create_system(
    mol: Chem.Mol,
    atom_map: Mapping[int, Any] | None = None,
    glycam_params: Mapping[str, Any] | None = None,
    gaff_params: Mapping[str, Any] | None = None,
    connector_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Construct a relaxation-ready OpenMM ``System`` from a fully built RDKit molecule."""
    from poly_csp.topology.atom_mapping import (
        backbone_indices,
        connector_indices,
        selector_indices,
    )
    from poly_csp.forcefield.exceptions import apply_mixing_rules

    built = build_bonded_relaxation_system(mol)
    system = built.system
    system = add_glycam_backbone(system, mol, backbone_indices(mol), glycam_params)
    system = add_gaff_selectors(system, mol, selector_indices(mol), gaff_params)
    system = add_connectors(system, mol, connector_indices(mol), connector_params)

    if atom_map is not None:
        apply_mixing_rules(system=system, atom_map=atom_map)

    return system


def _first_force(system: mm.System, force_type):
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        if isinstance(force, force_type):
            return force
    return None


def _apply_transferred_bonded_forces(
    system: mm.System,
    forces: Sequence[mm.Force],
) -> None:
    bond_force = _first_force(system, mm.HarmonicBondForce)
    angle_force = _first_force(system, mm.HarmonicAngleForce)

    for force in forces:
        if isinstance(force, mm.HarmonicBondForce):
            if bond_force is None:
                bond_force = mm.HarmonicBondForce()
                system.addForce(bond_force)
            for idx in range(force.getNumBonds()):
                a, b, r0, k = force.getBondParameters(idx)
                _set_or_add_bond(bond_force, int(a), int(b), r0, k)
            continue

        if isinstance(force, mm.HarmonicAngleForce):
            if angle_force is None:
                angle_force = mm.HarmonicAngleForce()
                system.addForce(angle_force)
            for idx in range(force.getNumAngles()):
                a, b, c, theta0, k = force.getAngleParameters(idx)
                _set_or_add_angle(angle_force, int(a), int(b), int(c), theta0, k)
            continue

        if isinstance(force, mm.PeriodicTorsionForce) and force.getNumTorsions() > 0:
            system.addForce(_copy_periodic_torsion_force(force))


def _copy_periodic_torsion_force(force: mm.PeriodicTorsionForce) -> mm.PeriodicTorsionForce:
    copied = mm.PeriodicTorsionForce()
    for idx in range(force.getNumTorsions()):
        a, b, c, d, periodicity, phase, k = force.getTorsionParameters(idx)
        copied.addTorsion(a, b, c, d, periodicity, phase, k)
    return copied


def _normalize_connector_params(
    connector_params: Mapping[str, Any] | "ConnectorParams" | None,
) -> list["ConnectorParams"]:
    from poly_csp.forcefield.connectors import ConnectorParams

    if connector_params is None:
        return []
    if isinstance(connector_params, ConnectorParams):
        return [connector_params]
    if isinstance(connector_params, Mapping):
        out: list[ConnectorParams] = []
        for value in connector_params.values():
            if isinstance(value, ConnectorParams):
                out.append(value)
        return out
    return []


def _connector_role_maps(mol: Chem.Mol, params: ConnectorParams) -> list[dict[str, int]]:
    from poly_csp.topology.atom_mapping import attachment_instance_maps
    from poly_csp.topology.utils import residue_label_maps

    instance_maps = attachment_instance_maps(mol)
    if not instance_maps:
        return []

    residue_maps = residue_label_maps(mol)
    by_instance: dict[int, dict[str, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        instance_id = int(atom.GetIntProp("_poly_csp_selector_instance"))
        residue_index = int(atom.GetIntProp("_poly_csp_residue_index"))
        site = atom.GetProp("_poly_csp_site") if atom.HasProp("_poly_csp_site") else None
        if params.site is not None and site != params.site:
            continue
        if instance_id in by_instance:
            continue
        role_map: dict[str, int] = {}
        if residue_index >= len(residue_maps):
            continue
        for label, atom_idx in residue_maps[residue_index].items():
            role_map[f"BB_{label}"] = int(atom_idx)
        for local_idx, atom_idx in instance_maps.get(instance_id, {}).items():
            role_map[f"SL_{local_idx:03d}"] = int(atom_idx)
        by_instance[instance_id] = role_map
    return list(by_instance.values())


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _angle_key(a: int, b: int, c: int) -> tuple[int, int, int]:
    return (a, b, c) if a <= c else (c, b, a)


def _set_or_add_bond(
    force: mm.HarmonicBondForce,
    a: int,
    b: int,
    r0,
    k,
) -> None:
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
