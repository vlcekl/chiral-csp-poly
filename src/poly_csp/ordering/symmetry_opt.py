from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from rdkit import Chem

from poly_csp.chemistry.functionalization import apply_selector_pose_dihedrals
from poly_csp.chemistry.selectors import SelectorTemplate
from poly_csp.config.schema import Site
from poly_csp.ordering.hbonds import HbondMetrics, compute_hbond_metrics
from poly_csp.ordering.rotamers import (
    RotamerGridSpec,
    default_rotamer_grid,
    enumerate_pose_library,
)
from poly_csp.ordering.scoring import (
    bonded_exclusion_pairs,
    min_distance_by_class,
    min_distance_by_class_fast,
    min_interatomic_distance,
    min_interatomic_distance_fast,
)


@dataclass(frozen=True)
class OrderingSpec:
    enabled: bool = False
    repeat_residues: int = 1
    max_distance_A: float = 3.3
    neighbor_window: int = 1
    min_donor_angle_deg: float = 100.0
    min_acceptor_angle_deg: float = 90.0
    exclude_13: bool = True
    exclude_14: bool = False
    min_backbone_backbone_distance_A: float = 1.05
    min_backbone_selector_distance_A: float = 1.0
    min_selector_selector_distance_A: float = 1.0
    hbond_weight: float = 8.0
    clash_weight: float = 10.0
    max_candidates: int = 64


def _heavy_mask(mol: Chem.Mol) -> np.ndarray:
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _objective(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    spec: OrderingSpec,
    use_fast: bool = True,
) -> tuple[float, HbondMetrics, float, dict[str, float]]:
    hb = compute_hbond_metrics(
        mol=mol,
        selector=selector,
        max_distance_A=spec.max_distance_A,
        neighbor_window=spec.neighbor_window,
        min_donor_angle_deg=spec.min_donor_angle_deg,
        min_acceptor_angle_deg=spec.min_acceptor_angle_deg,
    )
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    max_path_length = 1 + int(spec.exclude_13) + int(spec.exclude_14)
    excluded = bonded_exclusion_pairs(mol, max_path_length=max_path_length)
    heavy_mask = _heavy_mask(mol)

    if use_fast:
        dmin = float(min_interatomic_distance_fast(xyz, heavy_mask, excluded))
        class_min = min_distance_by_class_fast(mol, xyz, heavy_mask, excluded)
    else:
        dmin = float(min_interatomic_distance(xyz, heavy_mask, excluded))
        class_min = min_distance_by_class(mol, xyz, heavy_mask, excluded)

    bb = class_min["backbone_backbone"]
    bs = class_min["backbone_selector"]
    ss = class_min["selector_selector"]
    deficit = 0.0
    if np.isfinite(bb):
        deficit += max(0.0, float(spec.min_backbone_backbone_distance_A) - float(bb))
    if np.isfinite(bs):
        deficit += max(0.0, float(spec.min_backbone_selector_distance_A) - float(bs))
    if np.isfinite(ss):
        deficit += max(0.0, float(spec.min_selector_selector_distance_A) - float(ss))

    hbond_component = float(spec.hbond_weight) * float(hb.geometric_fraction)
    clash_component = float(spec.clash_weight) * float(deficit)
    score = hbond_component - clash_component
    return score, hb, dmin, class_min


def optimize_selector_ordering(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    sites: Iterable[Site],
    dp: int,
    spec: OrderingSpec,
    grid: RotamerGridSpec | None = None,
    seed: int | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    """
    Deterministic stage-4 optimizer.

    When repeat_residues > 1, each residue within the repeat unit is
    optimized independently (coordinate descent), then the pattern is
    replicated periodically across the chain.
    """
    if not spec.enabled:
        baseline_score, baseline_hb, baseline_dmin, baseline_class = _objective(
            mol, selector, spec
        )
        return Chem.Mol(mol), {
            "enabled": False,
            "baseline_score": baseline_score,
            "baseline_hbond_like_fraction": baseline_hb.like_fraction,
            "baseline_hbond_geometric_fraction": baseline_hb.geometric_fraction,
            "baseline_min_heavy_distance_A": baseline_dmin,
            "baseline_class_min_distance_A": {
                k: _finite_or_none(v) for k, v in baseline_class.items()
            },
            "final_score": baseline_score,
            "final_hbond_like_fraction": baseline_hb.like_fraction,
            "final_hbond_geometric_fraction": baseline_hb.geometric_fraction,
            "final_min_heavy_distance_A": baseline_dmin,
            "final_class_min_distance_A": {
                k: _finite_or_none(v) for k, v in baseline_class.items()
            },
            "selected_pose_by_site": {},
        }

    grid_spec = grid or default_rotamer_grid(selector.name)
    if spec.max_candidates > 0:
        grid_spec = RotamerGridSpec(
            dihedral_values_deg=grid_spec.dihedral_values_deg,
            max_candidates=min(grid_spec.max_candidates, spec.max_candidates),
        )
    pose_library = enumerate_pose_library(grid_spec)

    # Shuffle pose traversal order for diversity across multi-start runs.
    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(pose_library))
        pose_library = [pose_library[i] for i in indices]

    work = Chem.Mol(mol)
    baseline_score, baseline_hb, baseline_dmin, baseline_class = _objective(
        work, selector, spec
    )
    selected: Dict[str, Dict[int, Dict[str, float]]] = {}

    repeat = max(1, min(int(spec.repeat_residues), int(dp)))
    residues = list(range(int(dp)))

    for site in [str(s) for s in sites]:
        # Per-residue coordinate descent within the repeat unit.
        # Each residue in the repeat unit gets its own best pose.
        per_residue_poses: Dict[int, Dict[str, float]] = {}

        # Initialize: no dihedral changes (current state)
        for r in range(repeat):
            per_residue_poses[r] = {}

        max_cd_cycles = 3  # coordinate descent iterations
        for _cycle in range(max_cd_cycles):
            improved = False
            for res_in_repeat in range(repeat):
                best_mol = Chem.Mol(work)
                best_score = float("-inf")
                best_pose: Dict[str, float] = dict(per_residue_poses[res_in_repeat])

                for pose in pose_library:
                    trial = Chem.Mol(work)
                    # Apply this candidate pose to all residues matching
                    # this position in the repeat unit.
                    for residue_index in residues:
                        if residue_index % repeat == res_in_repeat:
                            trial = apply_selector_pose_dihedrals(
                                mol=trial,
                                residue_index=residue_index,
                                site=site,  # type: ignore[arg-type]
                                pose_spec=pose,
                                selector=selector,
                            )
                    score, _, _, _ = _objective(trial, selector, spec)
                    if score > best_score:
                        best_score = score
                        best_mol = trial
                        best_pose = dict(pose.dihedral_targets_deg)
                        improved = True

                work = best_mol
                per_residue_poses[res_in_repeat] = best_pose

            if not improved:
                break  # converged

        selected[site] = per_residue_poses

    final_score, final_hb, final_dmin, final_class = _objective(work, selector, spec)

    # Flatten per-residue poses for summary
    selected_summary: Dict[str, object] = {}
    for site_key, residue_poses in selected.items():
        selected_summary[site_key] = {
            str(r): pose for r, pose in residue_poses.items()
        }

    summary: Dict[str, object] = {
        "enabled": True,
        "repeat_residues": repeat,
        "candidate_count": len(pose_library),
        "seed": seed,
        "baseline_score": baseline_score,
        "baseline_hbond_like_fraction": baseline_hb.like_fraction,
        "baseline_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "baseline_min_heavy_distance_A": baseline_dmin,
        "baseline_class_min_distance_A": {
            k: _finite_or_none(v) for k, v in baseline_class.items()
        },
        "final_score": final_score,
        "final_hbond_like_fraction": final_hb.like_fraction,
        "final_hbond_geometric_fraction": final_hb.geometric_fraction,
        "final_min_heavy_distance_A": final_dmin,
        "final_class_min_distance_A": {
            k: _finite_or_none(v) for k, v in final_class.items()
        },
        "selected_pose_by_site": selected_summary,
    }
    return work, summary
