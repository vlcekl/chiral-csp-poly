from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
from rdkit import Chem

from poly_csp.config.schema import Site
from poly_csp.forcefield.minimization import (
    PreparedRuntimeOptimizationBundle,
    RuntimeRestraintSpec,
    TwoStageMinimizationProtocol,
    positions_nm_from_mol,
    prepare_runtime_optimization_bundle,
    run_prepared_runtime_optimization,
    update_rdkit_coords,
)
from poly_csp.forcefield.runtime_params import RuntimeParams, load_runtime_params
from poly_csp.ordering.hbonds import HbondMetrics, compute_hbond_metrics
from poly_csp.ordering.rotamers import (
    RotamerGridSpec,
    default_rotamer_grid,
    enumerate_pose_library,
)
from poly_csp.ordering.scoring import (
    bonded_exclusion_pairs,
    min_distance_by_class_fast,
    min_interatomic_distance_fast,
)
from poly_csp.structure.alignment import apply_selector_pose_dihedrals
from poly_csp.structure.pbc import get_box_vectors_A
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class OrderingSpec:
    enabled: bool = False
    repeat_residues: int = 4
    max_candidates: int = 64
    positional_k: float = 5000.0
    freeze_backbone: bool = True
    soft_n_stages: int = 3
    soft_max_iterations: int = 60
    full_max_iterations: int = 120
    final_restraint_factor: float = 0.15
    max_site_sweeps: int = 5
    randomize_initial_assignment: bool = True
    randomize_site_order: bool = True
    randomize_residue_order: bool = True
    randomize_pose_order: bool = True
    hbond_max_distance_A: float = 3.3
    hbond_neighbor_window: int = 1
    hbond_min_donor_angle_deg: float = 100.0
    hbond_min_acceptor_angle_deg: float = 90.0


@dataclass(frozen=True)
class RuntimeOrderingEvaluation:
    mol: Chem.Mol
    score: float
    final_energy_kj_mol: float
    stage1_energies_kj_mol: tuple[float, ...]
    stage2_energies_kj_mol: tuple[float, ...]
    hbond_metrics: HbondMetrics
    min_heavy_distance_A: float
    class_min_distance_A: dict[str, float]


def _require_forcefield_molecule(mol: Chem.Mol) -> None:
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Ordering requires a forcefield-domain molecule from "
            "build_forcefield_molecule()."
        )


def _heavy_mask(mol: Chem.Mol) -> np.ndarray:
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _copy_pose(pose: Mapping[str, float]) -> Dict[str, float]:
    return {str(name): float(value) for name, value in pose.items()}


def _ordering_diagnostics(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    spec: OrderingSpec,
) -> tuple[HbondMetrics, float, dict[str, float]]:
    box_vectors_A = get_box_vectors_A(mol)
    hb = compute_hbond_metrics(
        mol=mol,
        selector=selector,
        max_distance_A=spec.hbond_max_distance_A,
        neighbor_window=spec.hbond_neighbor_window,
        min_donor_angle_deg=spec.hbond_min_donor_angle_deg,
        min_acceptor_angle_deg=spec.hbond_min_acceptor_angle_deg,
        box_vectors_A=box_vectors_A,
    )
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    excluded = bonded_exclusion_pairs(mol, max_path_length=2)
    heavy_mask = _heavy_mask(mol)
    dmin = float(
        min_interatomic_distance_fast(
            xyz,
            heavy_mask,
            excluded,
            box_vectors_A=box_vectors_A,
        )
    )
    class_min = min_distance_by_class_fast(
        mol,
        xyz,
        heavy_mask,
        excluded,
        box_vectors_A=box_vectors_A,
    )
    return hb, dmin, class_min


def _prepare_runtime_ordering_systems(
    mol: Chem.Mol,
    *,
    runtime_params: RuntimeParams,
    spec: OrderingSpec,
    mixing_rules_cfg: Mapping[str, object] | None,
) -> PreparedRuntimeOptimizationBundle:
    return prepare_runtime_optimization_bundle(
        mol,
        runtime_params=runtime_params,
        selector=None,
        mixing_rules_cfg=mixing_rules_cfg,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=float(spec.positional_k),
            dihedral_k=0.0,
            hbond_k=0.0,
            freeze_backbone=bool(spec.freeze_backbone),
        ),
        protocol=TwoStageMinimizationProtocol(
            soft_n_stages=int(spec.soft_n_stages),
            soft_max_iterations=int(spec.soft_max_iterations),
            full_max_iterations=int(spec.full_max_iterations),
            final_restraint_factor=float(spec.final_restraint_factor),
        ),
    )


def _evaluate_runtime_candidate(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    prepared: PreparedRuntimeOptimizationBundle,
    spec: OrderingSpec,
) -> RuntimeOrderingEvaluation:
    minimization = run_prepared_runtime_optimization(
        prepared,
        initial_positions_nm=positions_nm_from_mol(mol),
    )
    minimized = update_rdkit_coords(mol, minimization.final_positions_nm)
    hb, dmin, class_min = _ordering_diagnostics(minimized, selector, spec)
    final_energy = float(minimization.stage2_energies_kj_mol[-1])
    return RuntimeOrderingEvaluation(
        mol=minimized,
        score=-final_energy,
        final_energy_kj_mol=final_energy,
        stage1_energies_kj_mol=tuple(float(x) for x in minimization.stage1_energies_kj_mol),
        stage2_energies_kj_mol=tuple(float(x) for x in minimization.stage2_energies_kj_mol),
        hbond_metrics=hb,
        min_heavy_distance_A=dmin,
        class_min_distance_A={key: float(value) for key, value in class_min.items()},
    )


def optimize_selector_ordering(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    sites: Iterable[Site],
    dp: int,
    spec: OrderingSpec,
    grid: RotamerGridSpec | None = None,
    seed: int | None = None,
    *,
    runtime_params: RuntimeParams | None = None,
    work_dir: str | Path | None = None,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = None,
    mixing_rules_cfg: Mapping[str, object] | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    """
    Seeded selector ordering on the canonical all-atom forcefield molecule.

    Candidates are evaluated by short two-stage minimization on shared soft/full
    runtime systems. When a seed is provided, the search uses a randomized
    repeat-class initialization and randomized sweep order before greedy
    refinement. Final ranking is by the stage-2 full-system potential energy.
    """
    _require_forcefield_molecule(mol)

    if runtime_params is None:
        runtime_params = load_runtime_params(
            mol,
            selector_template=selector,
            work_dir=None if work_dir is None else Path(work_dir),
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

    if not spec.enabled:
        hb, dmin, class_min = _ordering_diagnostics(mol, selector, spec)
        return Chem.Mol(mol), {
            "enabled": False,
            "objective": "negative_stage2_energy_kj_mol",
            "baseline_energy_kj_mol": None,
            "baseline_hbond_like_fraction": hb.like_fraction,
            "baseline_hbond_geometric_fraction": hb.geometric_fraction,
            "baseline_hbond_donor_count": hb.donor_count,
            "baseline_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "baseline_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "baseline_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "baseline_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "baseline_min_heavy_distance_A": dmin,
            "baseline_class_min_distance_A": {
                k: _finite_or_none(v) for k, v in class_min.items()
            },
            "initialization_mode": "disabled",
            "site_sweep_count": 0,
            "initial_energy_kj_mol": None,
            "initial_hbond_like_fraction": hb.like_fraction,
            "initial_hbond_geometric_fraction": hb.geometric_fraction,
            "initial_hbond_donor_count": hb.donor_count,
            "initial_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "initial_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "initial_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "initial_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "final_energy_kj_mol": None,
            "final_score": None,
            "final_hbond_like_fraction": hb.like_fraction,
            "final_hbond_geometric_fraction": hb.geometric_fraction,
            "final_hbond_donor_count": hb.donor_count,
            "final_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "final_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "final_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "final_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "final_min_heavy_distance_A": dmin,
            "final_class_min_distance_A": {
                k: _finite_or_none(v) for k, v in class_min.items()
            },
            "initial_pose_by_site": {},
            "selected_pose_by_site": {},
        }

    prepared = _prepare_runtime_ordering_systems(
        mol,
        runtime_params=runtime_params,
        spec=spec,
        mixing_rules_cfg=mixing_rules_cfg,
    )

    grid_spec = grid or default_rotamer_grid(selector)
    if spec.max_candidates > 0:
        grid_spec = RotamerGridSpec(
            dihedral_values_deg=grid_spec.dihedral_values_deg,
            max_candidates=min(grid_spec.max_candidates, int(spec.max_candidates)),
        )
    pose_library = enumerate_pose_library(grid_spec)
    rng = np.random.default_rng(seed) if seed is not None else None
    if rng is not None:
        order = rng.permutation(len(pose_library))
        pose_library = [pose_library[int(i)] for i in order]

    baseline = _evaluate_runtime_candidate(
        Chem.Mol(mol),
        selector=selector,
        prepared=prepared,
        spec=spec,
    )

    repeat = max(1, min(int(spec.repeat_residues), int(dp)))
    residues = list(range(int(dp)))
    site_keys = [str(site) for site in sites]
    selected: Dict[str, Dict[int, Dict[str, float]]] = {
        site: {residue_in_repeat: {} for residue_in_repeat in range(repeat)}
        for site in site_keys
    }
    initial_pose_by_site: Dict[str, Dict[int, Dict[str, float]]] = {
        site: {residue_in_repeat: {} for residue_in_repeat in range(repeat)}
        for site in site_keys
    }
    evaluation_count = 1
    initialization_mode = "baseline_minimized"
    work = baseline.mol
    current = baseline

    if bool(spec.randomize_initial_assignment) and rng is not None and pose_library:
        randomized = Chem.Mol(baseline.mol)
        site_iter = list(site_keys)
        if bool(spec.randomize_site_order):
            rng.shuffle(site_iter)
        for site in site_iter:
            residue_iter = list(range(repeat))
            if bool(spec.randomize_residue_order):
                rng.shuffle(residue_iter)
            for residue_in_repeat in residue_iter:
                pose = pose_library[int(rng.integers(len(pose_library)))]
                pose_payload = _copy_pose(pose.dihedral_targets_deg)
                for residue_index in residues:
                    if residue_index % repeat != residue_in_repeat:
                        continue
                    randomized = apply_selector_pose_dihedrals(
                        mol=randomized,
                        residue_index=residue_index,
                        site=site,  # type: ignore[arg-type]
                        pose_spec=pose,
                        selector=selector,
                    )
                selected[site][residue_in_repeat] = dict(pose_payload)
                initial_pose_by_site[site][residue_in_repeat] = dict(pose_payload)
        current = _evaluate_runtime_candidate(
            randomized,
            selector=selector,
            prepared=prepared,
            spec=spec,
        )
        evaluation_count += 1
        work = current.mol
        initialization_mode = "seeded_random_assignment"

    initial = current

    sweep_count = 0
    for _ in range(max(1, int(spec.max_site_sweeps))):
        improved = False
        sweep_count += 1
        site_iter = list(site_keys)
        if rng is not None and bool(spec.randomize_site_order):
            rng.shuffle(site_iter)
        for site in site_iter:
            per_residue_poses = selected[site]
            residue_iter = list(range(repeat))
            if rng is not None and bool(spec.randomize_residue_order):
                rng.shuffle(residue_iter)
            for residue_in_repeat in residue_iter:
                best_eval = current
                best_pose = dict(per_residue_poses[residue_in_repeat])
                candidate_poses = pose_library
                if rng is not None and bool(spec.randomize_pose_order):
                    pose_order = rng.permutation(len(pose_library))
                    candidate_poses = [pose_library[int(i)] for i in pose_order]
                for pose in candidate_poses:
                    pose_payload = _copy_pose(pose.dihedral_targets_deg)
                    if pose_payload == best_pose:
                        continue
                    trial = Chem.Mol(work)
                    for residue_index in residues:
                        if residue_index % repeat != residue_in_repeat:
                            continue
                        trial = apply_selector_pose_dihedrals(
                            mol=trial,
                            residue_index=residue_index,
                            site=site,  # type: ignore[arg-type]
                            pose_spec=pose,
                            selector=selector,
                        )
                    trial_eval = _evaluate_runtime_candidate(
                        trial,
                        selector=selector,
                        prepared=prepared,
                        spec=spec,
                    )
                    evaluation_count += 1
                    if trial_eval.score > best_eval.score + 1e-9:
                        best_eval = trial_eval
                        best_pose = dict(pose_payload)
                if best_eval is not current:
                    work = best_eval.mol
                    current = best_eval
                    per_residue_poses[residue_in_repeat] = best_pose
                    improved = True
        if not improved:
            break

    final = current
    selected_summary: Dict[str, object] = {
        site_key: {str(residue): pose for residue, pose in residue_poses.items()}
        for site_key, residue_poses in selected.items()
    }
    initial_pose_summary: Dict[str, object] = {
        site_key: {str(residue): pose for residue, pose in residue_poses.items()}
        for site_key, residue_poses in initial_pose_by_site.items()
    }
    summary: Dict[str, object] = {
        "enabled": True,
        "objective": "negative_stage2_energy_kj_mol",
        "stage1_nonbonded_mode": prepared.soft.nonbonded_mode,
        "stage2_nonbonded_mode": prepared.full.nonbonded_mode,
        "repeat_residues": repeat,
        "candidate_count": len(pose_library),
        "evaluation_count": evaluation_count,
        "initialization_mode": initialization_mode,
        "site_sweep_count": sweep_count,
        "seed": seed,
        "baseline_energy_kj_mol": baseline.final_energy_kj_mol,
        "baseline_stage1_energies_kj_mol": list(baseline.stage1_energies_kj_mol),
        "baseline_stage2_energies_kj_mol": list(baseline.stage2_energies_kj_mol),
        "baseline_hbond_like_fraction": baseline.hbond_metrics.like_fraction,
        "baseline_hbond_geometric_fraction": baseline.hbond_metrics.geometric_fraction,
        "baseline_hbond_donor_count": baseline.hbond_metrics.donor_count,
        "baseline_hbond_like_satisfied_donors": baseline.hbond_metrics.like_satisfied_donors,
        "baseline_hbond_geometric_satisfied_donors": baseline.hbond_metrics.geometric_satisfied_donors,
        "baseline_hbond_like_donor_occupancy_fraction": (
            baseline.hbond_metrics.like_donor_occupancy_fraction
        ),
        "baseline_hbond_geometric_donor_occupancy_fraction": (
            baseline.hbond_metrics.geometric_donor_occupancy_fraction
        ),
        "baseline_min_heavy_distance_A": baseline.min_heavy_distance_A,
        "baseline_class_min_distance_A": {
            k: _finite_or_none(v) for k, v in baseline.class_min_distance_A.items()
        },
        "initial_energy_kj_mol": initial.final_energy_kj_mol,
        "initial_hbond_like_fraction": initial.hbond_metrics.like_fraction,
        "initial_hbond_geometric_fraction": initial.hbond_metrics.geometric_fraction,
        "initial_hbond_donor_count": initial.hbond_metrics.donor_count,
        "initial_hbond_like_satisfied_donors": initial.hbond_metrics.like_satisfied_donors,
        "initial_hbond_geometric_satisfied_donors": initial.hbond_metrics.geometric_satisfied_donors,
        "initial_hbond_like_donor_occupancy_fraction": (
            initial.hbond_metrics.like_donor_occupancy_fraction
        ),
        "initial_hbond_geometric_donor_occupancy_fraction": (
            initial.hbond_metrics.geometric_donor_occupancy_fraction
        ),
        "final_energy_kj_mol": final.final_energy_kj_mol,
        "final_score": final.score,
        "final_stage1_energies_kj_mol": list(final.stage1_energies_kj_mol),
        "final_stage2_energies_kj_mol": list(final.stage2_energies_kj_mol),
        "final_hbond_like_fraction": final.hbond_metrics.like_fraction,
        "final_hbond_geometric_fraction": final.hbond_metrics.geometric_fraction,
        "final_hbond_donor_count": final.hbond_metrics.donor_count,
        "final_hbond_like_satisfied_donors": final.hbond_metrics.like_satisfied_donors,
        "final_hbond_geometric_satisfied_donors": final.hbond_metrics.geometric_satisfied_donors,
        "final_hbond_like_donor_occupancy_fraction": (
            final.hbond_metrics.like_donor_occupancy_fraction
        ),
        "final_hbond_geometric_donor_occupancy_fraction": (
            final.hbond_metrics.geometric_donor_occupancy_fraction
        ),
        "final_min_heavy_distance_A": final.min_heavy_distance_A,
        "final_class_min_distance_A": {
            k: _finite_or_none(v) for k, v in final.class_min_distance_A.items()
        },
        "initial_pose_by_site": initial_pose_summary,
        "selected_pose_by_site": selected_summary,
    }
    return final.mol, summary
