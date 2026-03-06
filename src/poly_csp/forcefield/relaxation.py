from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import openmm as mm
from openmm import unit

from poly_csp.forcefield.connectors import parameterize_capped_monomer
from poly_csp.topology.atom_mapping import selector_instance_maps
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.structure.dihedrals import measure_dihedral_rad
from poly_csp.forcefield.anneal import run_heat_cool_cycle, run_temperature_ramp
from poly_csp.forcefield.system_builder import create_system
from poly_csp.forcefield.restraints import (
    add_dihedral_restraints,
    add_hbond_distance_restraints,
    add_positional_restraints,
)


@dataclass(frozen=True)
class RelaxSpec:
    enabled: bool
    positional_k: float
    dihedral_k: float
    hbond_k: float
    n_stages: int = 3
    max_iterations: int = 200
    freeze_backbone: bool = True
    anneal_enabled: bool = False
    t_start_K: float = 50.0
    t_end_K: float = 350.0
    anneal_steps: int = 2000
    anneal_cool_down: bool = True


# ---------------------------------------------------------------------------
# Atom classification helpers
# ---------------------------------------------------------------------------

def _backbone_heavy_indices(mol: Chem.Mol) -> list[int]:
    idx: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.HasProp("_poly_csp_selector_instance"):
            continue
        idx.append(atom.GetIdx())
    return idx


def _backbone_all_indices(mol: Chem.Mol) -> list[int]:
    """All backbone atom indices (heavy + hydrogen), preserving order."""
    idx: list[int] = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_poly_csp_selector_instance"):
            continue
        idx.append(atom.GetIdx())
    return idx


def _selector_all_indices(mol: Chem.Mol) -> set[int]:
    """Set of all selector atom indices."""
    return {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_selector_instance")
    }


def _selector_dihedral_targets(
    mol: Chem.Mol,
    selector: SelectorTemplate | None,
) -> list[tuple[int, int, int, int, float]]:
    if selector is None:
        return []
    if mol.GetNumConformers() == 0:
        return []
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    mappings = selector_instance_maps(mol)
    out: list[tuple[int, int, int, int, float]] = []
    for mapping in mappings.values():
        for _, (a_l, b_l, c_l, d_l) in selector.dihedrals.items():
            if selector.attach_dummy_idx is not None and selector.attach_dummy_idx in {
                a_l,
                b_l,
                c_l,
                d_l,
            }:
                continue
            if any(local not in mapping for local in (a_l, b_l, c_l, d_l)):
                continue
            a, b, c, d = mapping[a_l], mapping[b_l], mapping[c_l], mapping[d_l]
            theta0 = float(measure_dihedral_rad(xyz, a, b, c, d))
            out.append((a, b, c, d, theta0))
    return out


def _hbond_pairs(
    mol: Chem.Mol,
    selector: SelectorTemplate | None,
    max_dist_A: float = 3.3,
) -> list[tuple[int, int, float]]:
    if selector is None:
        return []
    if mol.GetNumConformers() == 0:
        return []
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    donors: list[tuple[int, int]] = []
    acceptors: list[tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        if local in selector.donors:
            donors.append((inst, atom.GetIdx()))
        if local in selector.acceptors:
            acceptors.append((inst, atom.GetIdx()))

    pairs: list[tuple[int, int, float]] = []
    max_dist_nm = float(max_dist_A / 10.0)
    for d_inst, d_idx in donors:
        for a_inst, a_idx in acceptors:
            if d_inst == a_inst:
                continue
            dist_nm = float(np.linalg.norm(xyz[d_idx] - xyz[a_idx]) / 10.0)
            if dist_nm <= max_dist_nm:
                pairs.append((d_idx, a_idx, dist_nm))
    return pairs


def _positions_nm_from_mol(mol: Chem.Mol) -> unit.Quantity:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before OpenMM system build.")
    xyz_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    return (xyz_A / 10.0) * unit.nanometer


def _selector_sites(mol: Chem.Mol) -> list[str]:
    sites = {
        atom.GetProp("_poly_csp_site")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_selector_instance") and atom.HasProp("_poly_csp_site")
    }
    return sorted(sites)


def _has_connector_atoms(mol: Chem.Mol) -> bool:
    return any(atom.HasProp("_poly_csp_connector_atom") for atom in mol.GetAtoms())


def _connector_params_by_site(
    mol: Chem.Mol,
    selector: SelectorTemplate | None,
    selector_prmtop_path: str | None,
) -> Dict[str, object] | None:
    import logging

    log = logging.getLogger(__name__)

    if selector is None or selector_prmtop_path is None:
        return None
    if not selector.connector_local_roles or not _has_connector_atoms(mol):
        return None
    if not mol.HasProp("_poly_csp_polymer"):
        log.warning(
            "Skipping connector parameterization because _poly_csp_polymer metadata is missing."
        )
        return None

    polymer = mol.GetProp("_poly_csp_polymer")
    params_by_site: Dict[str, object] = {}
    for site in _selector_sites(mol):
        params_by_site[site] = parameterize_capped_monomer(
            polymer=polymer,  # type: ignore[arg-type]
            selector_template=selector,
            site=site,  # type: ignore[arg-type]
        )
    return params_by_site or None


def _update_rdkit_coords(mol: Chem.Mol, positions_nm: unit.Quantity) -> Chem.Mol:
    xyz_A = positions_nm.value_in_unit(unit.nanometer) * 10.0
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(xyz_A):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_staged_relaxation(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
    amber_artifacts: Dict[str, object] | None = None,
    selector_prmtop_path: str | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    if not spec.enabled:
        return Chem.Mol(mol), {"enabled": False}

    return _run_hybrid_pre_relax(
        mol=mol, spec=spec, selector=selector,
        selector_prmtop_path=selector_prmtop_path,
    )


def _run_hybrid_pre_relax(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
    selector_prmtop_path: str | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    """Relaxation with a shared modular forcefield and optional frozen backbone.

    The base system always starts from the generic RDKit-derived bonded model.
    When ``selector_prmtop_path`` is provided, pure selector-core terms are
    overlaid from the isolated selector GAFF2 prmtop and connector terms are
    overlaid from capped-monomer parameterization. Backbone heavy atoms can be
    frozen (mass = 0) so only selectors and linkers explore conformational
    space during restrained minimization and annealing.
    """
    import logging

    log = logging.getLogger(__name__)

    # --- Classify atoms. ---
    backbone_heavy = _backbone_heavy_indices(mol)
    backbone_all = _backbone_all_indices(mol)
    selector_indices = _selector_all_indices(mol)

    use_gaff2 = selector_prmtop_path is not None and selector is not None and len(selector_indices) > 0
    gaff_params = None
    if use_gaff2:
        gaff_params = {
            "selector_prmtop_path": selector_prmtop_path,
            "selector_template": selector,
        }
    connector_params = _connector_params_by_site(
        mol=mol,
        selector=selector,
        selector_prmtop_path=selector_prmtop_path,
    )

    system = create_system(
        mol=mol,
        gaff_params=gaff_params,
        connector_params=connector_params,
    )
    positions_nm = _positions_nm_from_mol(mol)

    # --- Freeze backbone heavy atoms. ---
    #     mass=0 makes them immovable in both minimization and dynamics.
    if spec.freeze_backbone:
        for idx in backbone_heavy:
            system.setParticleMass(idx, 0.0)

    # --- Add restraints (selectors only benefit from these). ---
    pos_force = add_positional_restraints(
        system=system,
        atom_indices=backbone_heavy,
        reference_positions_nm=positions_nm,
        k_kj_per_mol_nm2=float(spec.positional_k),
    )
    tors_force = add_dihedral_restraints(
        system=system,
        dihedrals=_selector_dihedral_targets(mol, selector),
        k_kj_per_mol=float(spec.dihedral_k),
    )
    hb_force = add_hbond_distance_restraints(
        system=system,
        pairs=_hbond_pairs(mol, selector),
        k_kj_per_mol_nm2=float(spec.hbond_k),
    )

    # --- Simulation. ---
    integrator = mm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    context = mm.Context(system, integrator)
    context.setPositions(positions_nm)

    # --- Staged minimization: release selector restraints gradually. ---
    stage_factors = np.linspace(1.0, 0.15, max(1, int(spec.n_stages)))
    stage_energies: list[float] = []
    for factor in stage_factors:
        context.setParameter("k_pos", float(spec.positional_k) * float(factor))
        context.setParameter("k_tors", float(spec.dihedral_k) * float(factor))
        context.setParameter("k_hb", float(spec.hbond_k) * float(factor))
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    # --- Annealing with frozen backbone. ---
    if spec.anneal_enabled and int(spec.anneal_steps) > 0:
        if spec.anneal_cool_down:
            run_heat_cool_cycle(
                context=context,
                integrator=integrator,
                t_start_K=float(spec.t_start_K),
                t_peak_K=float(spec.t_end_K),
                n_steps=int(spec.anneal_steps),
                n_segments=10,
            )
        else:
            run_temperature_ramp(
                context=context,
                integrator=integrator,
                t_start_K=float(spec.t_start_K),
                t_end_K=float(spec.t_end_K),
                n_steps=int(spec.anneal_steps),
                n_segments=10,
            )
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    final_state = context.getState(getPositions=True, getEnergy=True)
    final_positions = final_state.getPositions(asNumpy=True)

    # Sanity check: detect structure explosion.
    final_xyz_A = np.asarray(
        final_positions.value_in_unit(unit.nanometer)
    ) * 10.0
    span = final_xyz_A.max(axis=0) - final_xyz_A.min(axis=0)
    if np.any(span > 500.0):
        log.warning(
            "Relaxed structure has excessive span (%.0f x %.0f x %.0f Å). "
            "Bonded forces may be insufficient.",
            *span,
        )

    # Verify backbone didn't move.
    if spec.freeze_backbone:
        init_backbone_xyz = np.asarray(
            positions_nm.value_in_unit(unit.nanometer)
        )[backbone_heavy] * 10.0
        final_backbone_xyz = final_xyz_A[backbone_heavy]
        backbone_drift = np.max(np.abs(final_backbone_xyz - init_backbone_xyz))
        if backbone_drift > 0.01:
            log.warning(
                "Backbone drifted by %.4f Å despite freeze_backbone=True.",
                backbone_drift,
            )

    out = _update_rdkit_coords(mol, final_positions)
    summary: Dict[str, object] = {
        "enabled": True,
        "force_model": (
            "modular_gaff2_connectors"
            if use_gaff2 and connector_params
            else ("gaff2_selectors" if use_gaff2 else "hybrid_frozen_backbone")
        ),
        "n_stages": int(spec.n_stages),
        "stage_energies_kj_mol": stage_energies,
        "anneal_enabled": bool(spec.anneal_enabled),
        "anneal_cool_down": bool(spec.anneal_cool_down),
        "freeze_backbone": bool(spec.freeze_backbone),
        "n_backbone_atoms": len(backbone_all),
        "n_backbone_heavy_frozen": len(backbone_heavy),
        "n_selector_atoms": len(selector_indices),
    }
    return out, summary
