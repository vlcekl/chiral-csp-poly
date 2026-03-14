from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional

import numpy as np
from rdkit import Chem

from poly_csp.config.schema import (
    HbondPairingMode,
    HbondRestraintAtomMode,
    Site,
    SoftSelectorHbondBiasOptions,
)
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
from poly_csp.ordering.hbonds import (
    HbondMetrics,
    HbondConnectivityPolicy,
    SelectorHbondDiagnostics,
    compute_selector_hbond_diagnostics,
    resolve_hbond_connectivity_policy,
)
from poly_csp.ordering.rotamers import (
    RotamerGridSpec,
    default_rotamer_grid,
    enumerate_pose_library,
)
from poly_csp.ordering.scoring import (
    bonded_exclusion_pairs,
    min_distance_by_class_fast,
    min_interatomic_distance_fast,
    selector_aromatic_stacking_metrics,
)
from poly_csp.structure.alignment import apply_selector_pose_dihedrals
from poly_csp.structure.pbc import get_box_vectors_A
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class OrderingSpec:
    enabled: bool = False
    strategy: Literal["greedy", "symmetry_coupled", "symmetry_network"] = "greedy"
    repeat_residues: Optional[int] = None
    max_candidates: int = 64
    positional_k: float = 5000.0
    freeze_backbone: bool = True
    soft_n_stages: int = 3
    soft_max_iterations: int = 60
    full_max_iterations: int = 120
    final_restraint_factor: float = 0.15
    soft_repulsion_k_kj_per_mol_nm2: float = 800.0
    soft_repulsion_cutoff_nm: float = 0.6
    hbond_k: float = 0.0
    anti_stacking_sigma_scale: float = 1.0
    soft_exclude_14: bool = False
    ideal_hbond_target_nm: float | None = None
    hbond_pairing_mode: HbondPairingMode = "legacy_all_pairs"
    hbond_restraint_atom_mode: HbondRestraintAtomMode = "hydrogen_if_present"
    skip_full_stage: bool = False
    soft_selector_hbond_bias: SoftSelectorHbondBiasOptions = field(
        default_factory=SoftSelectorHbondBiasOptions
    )
    max_site_sweeps: int = 5
    randomize_initial_assignment: bool = True
    randomize_site_order: bool = True
    randomize_residue_order: bool = True
    randomize_pose_order: bool = True
    hbond_max_distance_A: float = 3.3
    hbond_neighbor_window: int = 1
    hbond_min_donor_angle_deg: float = 100.0
    hbond_min_acceptor_angle_deg: float = 90.0
    hbond_connectivity_policy: HbondConnectivityPolicy = "auto"
    symmetry_maxiter: int = 60
    symmetry_popsize: int = 12
    symmetry_polish: bool = False
    symmetry_network_min_heavy_distance_A: float = 1.7
    symmetry_network_clash_penalty: float = 1.0e6
    symmetry_network_weight_geom_occ: float = 100.0
    symmetry_network_weight_like_occ: float = 20.0
    symmetry_network_weight_geom_frac: float = 5.0
    symmetry_network_weight_family_min_geom: float = 150.0
    symmetry_network_weight_family_min_like: float = 40.0
    symmetry_network_weight_soft_energy: float = 1.0
    symmetry_network_weight_full_energy: float = 1.0
    symmetry_network_energy_clip: float = 5.0
    symmetry_network_rerank_population: bool = True
    symmetry_network_use_full_energy_in_search: bool = True
    symmetry_init_from_rotamer_grid: bool = True
    symmetry_init_jitter_deg: float = 10.0


@dataclass(frozen=True)
class RuntimeOrderingEvaluation:
    mol: Chem.Mol
    score: float
    final_energy_kj_mol: float
    stage1_energies_kj_mol: tuple[float, ...]
    stage2_energies_kj_mol: tuple[float, ...]
    full_stage_skipped: bool
    soft_nonbonded_mode: str
    full_nonbonded_mode: str
    soft_exception_summary: dict[str, object]
    hbond_metrics: HbondMetrics
    min_heavy_distance_A: float
    class_min_distance_A: dict[str, float]
    selector_aromatic_stacking_A: dict[str, object]


def _ordering_objective_label(spec: OrderingSpec) -> str:
    if spec.strategy == "symmetry_network":
        return "negative_network_first_symmetry_score"
    if spec.strategy == "symmetry_coupled":
        return (
            "negative_stage1_single_point_energy_kj_mol"
            if bool(spec.skip_full_stage)
            else "negative_stage2_single_point_energy_kj_mol"
        )
    return (
        "negative_stage1_energy_kj_mol"
        if bool(spec.skip_full_stage)
        else "negative_stage2_energy_kj_mol"
    )


def _ordering_uses_dynamic_runtime_bundle(spec: OrderingSpec) -> bool:
    return float(spec.hbond_k) > 0.0


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


def _resolved_repeat_residues(mol: Chem.Mol, spec: OrderingSpec, dp: int) -> int:
    if spec.repeat_residues is not None:
        return max(1, min(int(spec.repeat_residues), int(dp)))
    if mol.HasProp("_poly_csp_helix_repeat_residues"):
        return max(1, min(int(mol.GetIntProp("_poly_csp_helix_repeat_residues")), int(dp)))
    return 1


def _ordering_diagnostics(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    spec: OrderingSpec,
) -> tuple[HbondMetrics, float, dict[str, float], dict[str, object]]:
    hb_diag = _ordering_hbond_diagnostics(mol, selector, spec)
    hb = hb_diag.metrics
    box_vectors_A = get_box_vectors_A(mol)
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
    stacking = selector_aromatic_stacking_metrics(
        mol,
        selector.mol,
    )
    return hb, dmin, class_min, stacking


def _ordering_hbond_diagnostics(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    spec: OrderingSpec,
) -> SelectorHbondDiagnostics:
    box_vectors_A = get_box_vectors_A(mol)
    return compute_selector_hbond_diagnostics(
        mol=mol,
        selector=selector,
        max_distance_A=spec.hbond_max_distance_A,
        neighbor_window=spec.hbond_neighbor_window,
        min_donor_angle_deg=spec.hbond_min_donor_angle_deg,
        min_acceptor_angle_deg=spec.hbond_min_acceptor_angle_deg,
        box_vectors_A=box_vectors_A,
        connectivity_policy=spec.hbond_connectivity_policy,
    )


def _prepare_runtime_ordering_systems(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    runtime_params: RuntimeParams,
    spec: OrderingSpec,
    mixing_rules_cfg: Mapping[str, object] | None,
) -> PreparedRuntimeOptimizationBundle:
    return prepare_runtime_optimization_bundle(
        mol,
        runtime_params=runtime_params,
        selector=selector,
        mixing_rules_cfg=mixing_rules_cfg,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=float(spec.positional_k),
            dihedral_k=0.0,
            hbond_k=float(spec.hbond_k),
            freeze_backbone=bool(spec.freeze_backbone),
        ),
        protocol=TwoStageMinimizationProtocol(
            soft_n_stages=int(spec.soft_n_stages),
            soft_max_iterations=int(spec.soft_max_iterations),
            full_max_iterations=int(spec.full_max_iterations),
            final_restraint_factor=float(spec.final_restraint_factor),
            skip_full_stage=bool(spec.skip_full_stage),
        ),
        soft_repulsion_k_kj_per_mol_nm2=float(spec.soft_repulsion_k_kj_per_mol_nm2),
        soft_repulsion_cutoff_nm=float(spec.soft_repulsion_cutoff_nm),
        soft_exclude_14=bool(spec.soft_exclude_14),
        anti_stacking_sigma_scale=float(spec.anti_stacking_sigma_scale),
        soft_selector_hbond_bias=spec.soft_selector_hbond_bias,
        hbond_max_distance_A=float(spec.hbond_max_distance_A),
        hbond_neighbor_window=int(spec.hbond_neighbor_window),
        hbond_pairing_mode=spec.hbond_pairing_mode,
        hbond_restraint_atom_mode=spec.hbond_restraint_atom_mode,
        ideal_hbond_target_nm=spec.ideal_hbond_target_nm,
    )


def _evaluate_runtime_candidate(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    prepared: PreparedRuntimeOptimizationBundle | None,
    runtime_params: RuntimeParams,
    spec: OrderingSpec,
    mixing_rules_cfg: Mapping[str, object] | None,
) -> RuntimeOrderingEvaluation:
    resolved_prepared = prepared
    if resolved_prepared is None:
        resolved_prepared = _prepare_runtime_ordering_systems(
            mol,
            selector=selector,
            runtime_params=runtime_params,
            spec=spec,
            mixing_rules_cfg=mixing_rules_cfg,
        )
    minimization = run_prepared_runtime_optimization(
        resolved_prepared,
        initial_positions_nm=positions_nm_from_mol(mol),
    )
    minimized = update_rdkit_coords(mol, minimization.final_positions_nm)
    hb, dmin, class_min, stacking = _ordering_diagnostics(minimized, selector, spec)
    final_energy = float(
        (
            minimization.stage1_energies_kj_mol[-1]
            if minimization.full_stage_skipped
            else minimization.stage2_energies_kj_mol[-1]
        )
    )
    return RuntimeOrderingEvaluation(
        mol=minimized,
        score=-final_energy,
        final_energy_kj_mol=final_energy,
        stage1_energies_kj_mol=tuple(float(x) for x in minimization.stage1_energies_kj_mol),
        stage2_energies_kj_mol=tuple(float(x) for x in minimization.stage2_energies_kj_mol),
        full_stage_skipped=bool(minimization.full_stage_skipped),
        soft_nonbonded_mode=str(resolved_prepared.soft.nonbonded_mode),
        full_nonbonded_mode=str(resolved_prepared.full.nonbonded_mode),
        soft_exception_summary=dict(resolved_prepared.soft.exception_summary),
        hbond_metrics=hb,
        min_heavy_distance_A=dmin,
        class_min_distance_A={key: float(value) for key, value in class_min.items()},
        selector_aromatic_stacking_A=dict(stacking),
    )


def _optimize_selector_ordering_greedy(
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

    Candidates are evaluated by short runtime minimization on shared or
    candidate-specific soft/full systems. When a seed is provided, the search
    uses a randomized repeat-class initialization and randomized sweep order
    before greedy refinement. Final ranking follows the actual final stage:
    the full-stage energy in normal mode or the soft-stage energy when the full
    stage is intentionally skipped.
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
        hb, dmin, class_min, stacking = _ordering_diagnostics(mol, selector, spec)
        applied_hbond_connectivity_policy = resolve_hbond_connectivity_policy(
            mol,
            selector,
            requested_policy=spec.hbond_connectivity_policy,
            requested_sites=sites,
        )
        return Chem.Mol(mol), {
            "enabled": False,
            "strategy": "greedy",
            "objective": _ordering_objective_label(spec),
            "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
            "hbond_connectivity_policy_applied": str(applied_hbond_connectivity_policy),
            "baseline_energy_kj_mol": None,
            "stage1_nonbonded_mode": None,
            "stage2_nonbonded_mode": None,
            "full_stage_skipped": bool(spec.skip_full_stage),
            "final_stage_nonbonded_mode": None,
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
            "baseline_selector_aromatic_stacking_A": dict(stacking),
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
            "initial_selector_aromatic_stacking_A": dict(stacking),
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
            "final_selector_aromatic_stacking_A": dict(stacking),
            "initial_pose_by_site": {},
            "selected_pose_by_site": {},
        }

    prepared = (
        None
        if _ordering_uses_dynamic_runtime_bundle(spec)
        else _prepare_runtime_ordering_systems(
            mol,
            selector=selector,
            runtime_params=runtime_params,
            spec=spec,
            mixing_rules_cfg=mixing_rules_cfg,
        )
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
        runtime_params=runtime_params,
        spec=spec,
        mixing_rules_cfg=mixing_rules_cfg,
    )

    repeat = _resolved_repeat_residues(mol, spec, dp)
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
            runtime_params=runtime_params,
            spec=spec,
            mixing_rules_cfg=mixing_rules_cfg,
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
                        runtime_params=runtime_params,
                        spec=spec,
                        mixing_rules_cfg=mixing_rules_cfg,
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
        "strategy": "greedy",
        "objective": _ordering_objective_label(spec),
        "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
        "hbond_connectivity_policy_applied": str(
            resolve_hbond_connectivity_policy(
                final.mol,
                selector,
                requested_policy=spec.hbond_connectivity_policy,
                requested_sites=site_keys,
            )
        ),
        "stage1_nonbonded_mode": final.soft_nonbonded_mode,
        "soft_exception_summary": dict(final.soft_exception_summary),
        "stage2_nonbonded_mode": (
            None if final.full_stage_skipped else final.full_nonbonded_mode
        ),
        "full_stage_skipped": bool(final.full_stage_skipped),
        "final_stage_nonbonded_mode": (
            final.soft_nonbonded_mode
            if final.full_stage_skipped
            else final.full_nonbonded_mode
        ),
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
        "baseline_selector_aromatic_stacking_A": dict(
            baseline.selector_aromatic_stacking_A
        ),
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
        "initial_selector_aromatic_stacking_A": dict(
            initial.selector_aromatic_stacking_A
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
        "final_selector_aromatic_stacking_A": dict(final.selector_aromatic_stacking_A),
        "initial_pose_by_site": initial_pose_summary,
        "selected_pose_by_site": selected_summary,
    }
    return final.mol, summary


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
    if spec.strategy == "symmetry_coupled":
        from poly_csp.ordering.symmetry_opt import optimize_symmetry_coupled_ordering

        return optimize_symmetry_coupled_ordering(
            mol=mol,
            selector=selector,
            sites=sites,
            dp=dp,
            spec=spec,
            grid=grid,
            seed=seed,
            runtime_params=runtime_params,
            work_dir=work_dir,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            mixing_rules_cfg=mixing_rules_cfg,
        )
    if spec.strategy == "symmetry_network":
        from poly_csp.ordering.symmetry_opt import optimize_symmetry_network_ordering

        return optimize_symmetry_network_ordering(
            mol=mol,
            selector=selector,
            sites=sites,
            dp=dp,
            spec=spec,
            grid=grid,
            seed=seed,
            runtime_params=runtime_params,
            work_dir=work_dir,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            mixing_rules_cfg=mixing_rules_cfg,
        )
    return _optimize_selector_ordering_greedy(
        mol=mol,
        selector=selector,
        sites=sites,
        dp=dp,
        spec=spec,
        grid=grid,
        seed=seed,
        runtime_params=runtime_params,
        work_dir=work_dir,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
        mixing_rules_cfg=mixing_rules_cfg,
    )
