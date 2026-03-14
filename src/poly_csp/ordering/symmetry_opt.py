from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
from openmm import unit
from rdkit import Chem
from scipy.optimize import differential_evolution

from poly_csp.config.schema import Site
from poly_csp.forcefield.minimization import (
    new_context,
    potential_energy_kj_mol,
    update_rdkit_coords,
)
from poly_csp.forcefield.runtime_params import RuntimeParams, load_runtime_params
from poly_csp.ordering.hbonds import HbondMetrics, resolve_hbond_connectivity_policy
from poly_csp.ordering.optimize import (
    OrderingSpec,
    _ordering_hbond_diagnostics,
    _ordering_diagnostics,
    _ordering_objective_label,
    _prepare_runtime_ordering_systems,
    _require_forcefield_molecule,
)
from poly_csp.ordering.rotamers import (
    RotamerGridSpec,
    default_rotamer_grid,
)
from poly_csp.ordering.scoring import selector_screw_symmetry_rmsd_from_mol
from poly_csp.structure.alignment import (
    _downstream_mask,
    _resolve_selector_dihedral_atom_indices,
)
from poly_csp.structure.dihedrals import measure_dihedral_rad, set_dihedral_rad
from poly_csp.structure.matrix import ScrewTransform
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class _SiteBlock:
    site: str
    local_indices: tuple[int, ...]
    reference_indices: tuple[int, ...]
    residue_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _ActiveDihedral:
    site: str
    name: str
    atom_indices: tuple[int, int, int, int]
    rotate_mask: np.ndarray
    grid_values_deg: tuple[float, ...]


@dataclass
class _SymmetryOrderingEngine:
    mol: Chem.Mol
    selector: SelectorTemplate
    spec: OrderingSpec
    sites: tuple[str, ...]
    active_dihedrals: tuple[_ActiveDihedral, ...]
    site_blocks: tuple[_SiteBlock, ...]
    screw: ScrewTransform
    base_coords_A: np.ndarray
    soft_context: object
    soft_integrator: object
    full_context: object | None
    full_integrator: object | None
    soft_nonbonded_mode: str
    full_nonbonded_mode: str | None
    soft_exception_summary: dict[str, object]

    def initial_dihedrals_deg(self) -> dict[str, dict[str, float]]:
        return self.measure_dihedrals_deg(self.base_coords_A)

    def measure_dihedrals_deg(self, coords_A: np.ndarray) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {site: {} for site in self.sites}
        for term in self.active_dihedrals:
            angle_deg = float(
                np.rad2deg(measure_dihedral_rad(coords_A, *term.atom_indices))
            )
            out[term.site][term.name] = angle_deg
        return out

    def build_coords(self, values_deg: np.ndarray) -> np.ndarray:
        coords = np.asarray(self.base_coords_A, dtype=float).copy()
        for value_deg, term in zip(values_deg, self.active_dihedrals, strict=True):
            coords = set_dihedral_rad(
                coords=coords,
                a=term.atom_indices[0],
                b=term.atom_indices[1],
                c=term.atom_indices[2],
                d=term.atom_indices[3],
                target_angle_rad=np.deg2rad(float(value_deg)),
                rotate_mask=term.rotate_mask,
            )
        self._project_from_reference(coords)
        return coords

    def soft_energy_kj_mol(self, coords_A: np.ndarray) -> float:
        return self._energy_kj_mol(self.soft_context, coords_A)

    def full_energy_kj_mol(self, coords_A: np.ndarray) -> float:
        if self.full_context is None:
            return self.soft_energy_kj_mol(coords_A)
        return self._energy_kj_mol(self.full_context, coords_A)

    def _project_from_reference(self, coords_A: np.ndarray) -> None:
        for block in self.site_blocks:
            reference = coords_A[np.asarray(block.reference_indices, dtype=int)]
            for residue_index, target_indices in enumerate(block.residue_indices[1:], start=1):
                coords_A[np.asarray(target_indices, dtype=int)] = self.screw.apply(
                    reference,
                    residue_index,
                )

    @staticmethod
    def _energy_kj_mol(context, coords_A: np.ndarray) -> float:
        positions_nm = (np.asarray(coords_A, dtype=float) / 10.0) * unit.nanometer
        context.setPositions(positions_nm)
        return float(potential_energy_kj_mol(context))


def _resolved_screw_transform(mol: Chem.Mol) -> ScrewTransform:
    if not mol.HasProp("_poly_csp_helix_theta_rad") or not mol.HasProp("_poly_csp_helix_rise_A"):
        raise ValueError(
            "symmetry ordering requires _poly_csp_helix_theta_rad and "
            "_poly_csp_helix_rise_A metadata on the forcefield-domain molecule."
        )
    return ScrewTransform(
        theta_rad=float(mol.GetDoubleProp("_poly_csp_helix_theta_rad")),
        rise_A=float(mol.GetDoubleProp("_poly_csp_helix_rise_A")),
    )


def _site_attachment_local_to_global_map(
    mol: Chem.Mol,
    *,
    residue_index: int,
    site: str,
) -> dict[int, int]:
    instances: dict[int, dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        if not atom.HasProp("_poly_csp_residue_index") or not atom.HasProp("_poly_csp_site"):
            continue
        if int(atom.GetIntProp("_poly_csp_residue_index")) != int(residue_index):
            continue
        if atom.GetProp("_poly_csp_site") != str(site):
            continue
        instance = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local_idx = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        instances.setdefault(instance, {})[local_idx] = int(atom.GetIdx())
    if not instances:
        raise ValueError(
            f"No attached selector block found for residue {residue_index}, site {site!r}."
        )
    selected_instance = max(instances)
    return dict(instances[selected_instance])


def _site_block(
    mol: Chem.Mol,
    *,
    dp: int,
    site: str,
) -> _SiteBlock:
    reference = _site_attachment_local_to_global_map(mol, residue_index=0, site=site)
    local_indices = tuple(sorted(reference))
    residue_indices: list[tuple[int, ...]] = []
    for residue_index in range(int(dp)):
        mapping = _site_attachment_local_to_global_map(
            mol,
            residue_index=residue_index,
            site=site,
        )
        if tuple(sorted(mapping)) != local_indices:
            raise ValueError(
                f"Selector local-index mismatch for site {site!r} at residue {residue_index}."
            )
        residue_indices.append(tuple(mapping[local_idx] for local_idx in local_indices))
    return _SiteBlock(
        site=str(site),
        local_indices=local_indices,
        reference_indices=tuple(reference[local_idx] for local_idx in local_indices),
        residue_indices=tuple(residue_indices),
    )


def _active_dihedrals(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    sites: Iterable[str],
    active_values_by_name: Mapping[str, tuple[float, ...]],
) -> tuple[_ActiveDihedral, ...]:
    active: list[_ActiveDihedral] = []
    for site in sites:
        for name, grid_values in active_values_by_name.items():
            a, b, c, d = _resolve_selector_dihedral_atom_indices(
                mol,
                0,
                site,  # type: ignore[arg-type]
                selector,
                name,
            )
            active.append(
                _ActiveDihedral(
                    site=str(site),
                    name=str(name),
                    atom_indices=(int(a), int(b), int(c), int(d)),
                    rotate_mask=_downstream_mask(mol, int(b), int(c)),
                    grid_values_deg=tuple(float(value) for value in grid_values),
                )
            )
    return tuple(active)


def _active_grid_values(
    selector: SelectorTemplate,
    *,
    grid: RotamerGridSpec | None,
    include_anchor_dihedrals: bool,
) -> dict[str, tuple[float, ...]]:
    grid_spec = default_rotamer_grid(selector) if grid is None else grid
    values: dict[str, tuple[float, ...]] = {
        str(name): tuple(float(value) for value in dihedral_values)
        for name, dihedral_values in grid_spec.dihedral_values_deg.items()
    }
    if include_anchor_dihedrals:
        for name, dihedral_values in selector.anchor_rotamer_grid.items():
            values.setdefault(
                str(name),
                tuple(float(value) for value in dihedral_values),
            )
    return values


def _symmetry_pose_summary(
    values_by_site: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        site: {"0": {name: float(value) for name, value in sorted(dihedrals.items())}}
        for site, dihedrals in values_by_site.items()
    }


def _build_engine(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    sites: Iterable[str],
    dp: int,
    spec: OrderingSpec,
    grid: RotamerGridSpec | None,
    include_anchor_dihedrals: bool,
    runtime_params: RuntimeParams,
    mixing_rules_cfg: Mapping[str, object] | None,
) -> _SymmetryOrderingEngine:
    screw = _resolved_screw_transform(mol)
    site_list = tuple(str(site) for site in sites)
    site_blocks = tuple(
        _site_block(
            mol,
            dp=dp,
            site=site,
        )
        for site in site_list
    )

    active_values_by_name = _active_grid_values(
        selector,
        grid=grid,
        include_anchor_dihedrals=include_anchor_dihedrals,
    )
    active_dihedrals = _active_dihedrals(
        mol,
        selector=selector,
        sites=site_list,
        active_values_by_name={
            name: active_values_by_name[name]
            for name in sorted(active_values_by_name)
        },
    )

    raw_coords_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    base_coords_A = raw_coords_A.copy()
    for block in site_blocks:
        reference = raw_coords_A[np.asarray(block.reference_indices, dtype=int)]
        for residue_index, target_indices in enumerate(block.residue_indices[1:], start=1):
            base_coords_A[np.asarray(target_indices, dtype=int)] = screw.apply(
                reference,
                residue_index,
            )

    base_mol = update_rdkit_coords(mol, (base_coords_A / 10.0) * unit.nanometer)
    prepared = _prepare_runtime_ordering_systems(
        base_mol,
        selector=selector,
        runtime_params=runtime_params,
        spec=spec,
        mixing_rules_cfg=mixing_rules_cfg,
    )
    soft_context, soft_integrator = new_context(
        prepared.soft.system,
        prepared.soft.positions_nm,
    )
    full_context = None
    full_integrator = None
    if not bool(spec.skip_full_stage):
        full_context, full_integrator = new_context(
            prepared.full.system,
            prepared.full.positions_nm,
        )
    return _SymmetryOrderingEngine(
        mol=base_mol,
        selector=selector,
        spec=spec,
        sites=site_list,
        active_dihedrals=active_dihedrals,
        site_blocks=site_blocks,
        screw=screw,
        base_coords_A=base_coords_A,
        soft_context=soft_context,
        soft_integrator=soft_integrator,
        full_context=full_context,
        full_integrator=full_integrator,
        soft_nonbonded_mode=str(prepared.soft.nonbonded_mode),
        full_nonbonded_mode=(
            None if bool(spec.skip_full_stage) else str(prepared.full.nonbonded_mode)
        ),
        soft_exception_summary=dict(prepared.soft.exception_summary),
    )


@dataclass(frozen=True)
class _CandidateEval:
    values_deg: tuple[float, ...]
    coords_A: np.ndarray
    soft_energy_kj_mol: float
    full_energy_kj_mol: float
    score: float
    hbond_metrics: HbondMetrics
    hbond_like_fraction: float
    hbond_geometric_fraction: float
    hbond_like_donor_occupancy_fraction: float
    hbond_geometric_donor_occupancy_fraction: float
    hbond_like_satisfied_donors: int
    hbond_geometric_satisfied_donors: int
    hbond_family_metrics: dict[str, HbondMetrics]
    hbond_family_min_like_fraction: float
    hbond_family_min_geometric_fraction: float
    min_heavy_distance_A: float
    class_min_distance_A: dict[str, float | None]
    selector_aromatic_stacking_A: dict[str, object]


def _candidate_cache_key(values_deg: np.ndarray) -> tuple[float, ...]:
    arr = np.asarray(values_deg, dtype=float).reshape((-1,))
    return tuple(float(np.round(value, 6)) for value in arr)


def _relative_energy_term(
    energy_kj_mol: float,
    baseline_kj_mol: float,
    clip: float,
) -> float:
    scale = max(abs(float(baseline_kj_mol)), 1.0e3)
    term = (float(energy_kj_mol) - float(baseline_kj_mol)) / scale
    return float(np.clip(term, -abs(float(clip)), abs(float(clip))))


def _family_metric_min(
    family_metrics: Mapping[str, HbondMetrics],
    *,
    attr: str,
) -> float:
    if not family_metrics:
        return 1.0
    return min(float(getattr(metrics, attr)) for metrics in family_metrics.values())


def _hbond_family_metrics_summary(
    family_metrics: Mapping[str, HbondMetrics],
) -> dict[str, dict[str, float | int]]:
    return {
        str(name): {
            "like_fraction": float(metrics.like_fraction),
            "geometric_fraction": float(metrics.geometric_fraction),
            "donor_count": int(metrics.donor_count),
            "like_satisfied_pairs": int(metrics.like_satisfied_pairs),
            "geometric_satisfied_pairs": int(metrics.geometric_satisfied_pairs),
            "total_pairs": int(metrics.total_pairs),
            "like_donor_occupancy_fraction": float(metrics.like_donor_occupancy_fraction),
            "geometric_donor_occupancy_fraction": float(
                metrics.geometric_donor_occupancy_fraction
            ),
            "mean_like_distance_A": float(metrics.mean_like_distance_A),
            "mean_geometric_distance_A": float(metrics.mean_geometric_distance_A),
        }
        for name, metrics in sorted(family_metrics.items())
    }


def _initial_population_from_rotamers(
    engine: _SymmetryOrderingEngine,
    *,
    initial_values_deg: np.ndarray,
    popsize: int,
    jitter_deg: float,
    seed: int | None,
) -> np.ndarray | None:
    dims = int(initial_values_deg.size)
    if dims <= 0:
        return None

    size = max(5, int(popsize) * dims)
    rng = np.random.default_rng(seed)
    population = np.zeros((size, dims), dtype=float)
    population[0] = np.asarray(initial_values_deg, dtype=float)
    jitter = max(0.0, float(jitter_deg))

    for row_idx in range(1, size):
        for col_idx, term in enumerate(engine.active_dihedrals):
            values = (
                term.grid_values_deg
                if term.grid_values_deg
                else (float(initial_values_deg[col_idx]),)
            )
            base = float(values[int(rng.integers(len(values)))])
            offset = float(rng.uniform(-jitter, jitter)) if jitter > 0.0 else 0.0
            population[row_idx, col_idx] = ((base + offset + 180.0) % 360.0) - 180.0
    return population


def _evaluate_network_candidate(
    engine: _SymmetryOrderingEngine,
    *,
    selector: SelectorTemplate,
    values_deg: np.ndarray,
    baseline_soft_energy: float,
    baseline_full_energy: float,
    cache: dict[tuple[float, ...], _CandidateEval],
) -> _CandidateEval:
    key = _candidate_cache_key(values_deg)
    cached = cache.get(key)
    if cached is not None:
        return cached

    coords_A = engine.build_coords(np.asarray(values_deg, dtype=float))
    candidate_mol = update_rdkit_coords(engine.mol, (coords_A / 10.0) * unit.nanometer)
    hb_diag = _ordering_hbond_diagnostics(candidate_mol, selector, engine.spec)
    hb = hb_diag.metrics
    family_metrics = dict(hb_diag.family_metrics)
    family_min_like = _family_metric_min(
        family_metrics,
        attr="like_fraction",
    )
    family_min_geom = _family_metric_min(
        family_metrics,
        attr="geometric_fraction",
    )
    _, dmin, class_min, stacking = _ordering_diagnostics(candidate_mol, selector, engine.spec)

    clash_cutoff = float(engine.spec.symmetry_network_min_heavy_distance_A)
    clash_penalty = float(engine.spec.symmetry_network_clash_penalty)
    clipped_class_min = {
        key_name: (float(value) if np.isfinite(value) else None)
        for key_name, value in class_min.items()
    }
    if float(dmin) < clash_cutoff:
        delta = clash_cutoff - float(dmin)
        result = _CandidateEval(
            values_deg=tuple(float(value) for value in np.asarray(values_deg, dtype=float)),
            coords_A=coords_A,
            soft_energy_kj_mol=float("inf"),
            full_energy_kj_mol=float("inf"),
            score=float(clash_penalty) * (1.0 + delta * delta),
            hbond_metrics=hb,
            hbond_like_fraction=float(hb.like_fraction),
            hbond_geometric_fraction=float(hb.geometric_fraction),
            hbond_like_donor_occupancy_fraction=float(hb.like_donor_occupancy_fraction),
            hbond_geometric_donor_occupancy_fraction=float(
                hb.geometric_donor_occupancy_fraction
            ),
            hbond_like_satisfied_donors=int(hb.like_satisfied_donors),
            hbond_geometric_satisfied_donors=int(hb.geometric_satisfied_donors),
            hbond_family_metrics=family_metrics,
            hbond_family_min_like_fraction=float(family_min_like),
            hbond_family_min_geometric_fraction=float(family_min_geom),
            min_heavy_distance_A=float(dmin),
            class_min_distance_A=clipped_class_min,
            selector_aromatic_stacking_A=dict(stacking),
        )
        cache[key] = result
        return result

    soft_energy = float(engine.soft_energy_kj_mol(coords_A))
    use_full_energy = bool(
        engine.full_context is not None and engine.spec.symmetry_network_use_full_energy_in_search
    )
    full_energy = (
        float(engine.full_energy_kj_mol(coords_A))
        if use_full_energy
        else float(soft_energy)
    )
    score = (
        float(engine.spec.symmetry_network_weight_geom_occ)
        * (1.0 - float(hb.geometric_donor_occupancy_fraction))
        + float(engine.spec.symmetry_network_weight_like_occ)
        * (1.0 - float(hb.like_donor_occupancy_fraction))
        + float(engine.spec.symmetry_network_weight_geom_frac)
        * (1.0 - float(hb.geometric_fraction))
        + float(engine.spec.symmetry_network_weight_family_min_geom)
        * (1.0 - float(family_min_geom))
        + float(engine.spec.symmetry_network_weight_family_min_like)
        * (1.0 - float(family_min_like))
        + float(engine.spec.symmetry_network_weight_soft_energy)
        * _relative_energy_term(
            soft_energy,
            baseline_soft_energy,
            float(engine.spec.symmetry_network_energy_clip),
        )
        + float(engine.spec.symmetry_network_weight_full_energy)
        * _relative_energy_term(
            full_energy,
            baseline_full_energy,
            float(engine.spec.symmetry_network_energy_clip),
        )
    )
    result = _CandidateEval(
        values_deg=tuple(float(value) for value in np.asarray(values_deg, dtype=float)),
        coords_A=coords_A,
        soft_energy_kj_mol=float(soft_energy),
        full_energy_kj_mol=float(full_energy),
        score=float(score),
        hbond_metrics=hb,
        hbond_like_fraction=float(hb.like_fraction),
        hbond_geometric_fraction=float(hb.geometric_fraction),
        hbond_like_donor_occupancy_fraction=float(hb.like_donor_occupancy_fraction),
        hbond_geometric_donor_occupancy_fraction=float(
            hb.geometric_donor_occupancy_fraction
        ),
        hbond_like_satisfied_donors=int(hb.like_satisfied_donors),
        hbond_geometric_satisfied_donors=int(hb.geometric_satisfied_donors),
        hbond_family_metrics=family_metrics,
        hbond_family_min_like_fraction=float(family_min_like),
        hbond_family_min_geometric_fraction=float(family_min_geom),
        min_heavy_distance_A=float(dmin),
        class_min_distance_A=clipped_class_min,
        selector_aromatic_stacking_A=dict(stacking),
    )
    cache[key] = result
    return result


def _rank_network_candidates(candidates: Iterable[_CandidateEval]) -> _CandidateEval:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.hbond_family_min_geometric_fraction),
            -float(item.hbond_family_min_like_fraction),
            -float(item.hbond_geometric_donor_occupancy_fraction),
            -float(item.hbond_geometric_fraction),
            (
                float(item.hbond_metrics.mean_geometric_distance_A)
                if int(item.hbond_metrics.geometric_satisfied_pairs) > 0
                else float("inf")
            ),
            -float(item.hbond_like_donor_occupancy_fraction),
            -float(item.hbond_like_fraction),
            (
                float(item.hbond_metrics.mean_like_distance_A)
                if int(item.hbond_metrics.like_satisfied_pairs) > 0
                else float("inf")
            ),
            -float(item.min_heavy_distance_A),
            float(item.full_energy_kj_mol),
            float(item.soft_energy_kj_mol),
            float(item.score),
        ),
    )
    if not ranked:
        raise ValueError("No symmetry-network candidates available for ranking.")
    return ranked[0]


def optimize_symmetry_coupled_ordering(
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
        symmetry_rmsd = selector_screw_symmetry_rmsd_from_mol(mol)
        applied_hbond_connectivity_policy = resolve_hbond_connectivity_policy(
            mol,
            selector,
            requested_policy=spec.hbond_connectivity_policy,
            requested_sites=sites,
        )
        return Chem.Mol(mol), {
            "enabled": False,
            "strategy": "symmetry_coupled",
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
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "baseline_selector_aromatic_stacking_A": dict(stacking),
            "baseline_selector_symmetry_rmsd_A": float(symmetry_rmsd),
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
            "initial_selector_symmetry_rmsd_A": float(symmetry_rmsd),
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
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "final_selector_aromatic_stacking_A": dict(stacking),
            "final_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "initial_pose_by_site": {},
            "selected_pose_by_site": {},
        }

    engine = _build_engine(
        mol,
        selector=selector,
        sites=[str(site) for site in sites],
        dp=dp,
        spec=spec,
        grid=grid,
        include_anchor_dihedrals=False,
        runtime_params=runtime_params,
        mixing_rules_cfg=mixing_rules_cfg,
    )
    initial_pose = engine.initial_dihedrals_deg()
    initial_values_deg = np.asarray(
        [
            initial_pose[term.site][term.name]
            for term in engine.active_dihedrals
        ],
        dtype=float,
    )
    baseline_coords_A = engine.build_coords(initial_values_deg)
    baseline_mol = update_rdkit_coords(
        engine.mol,
        (baseline_coords_A / 10.0) * unit.nanometer,
    )
    baseline_soft_energy = engine.soft_energy_kj_mol(baseline_coords_A)
    baseline_full_energy = (
        baseline_soft_energy
        if bool(spec.skip_full_stage)
        else engine.full_energy_kj_mol(baseline_coords_A)
    )
    best_values_deg = initial_values_deg.copy()
    best_coords_A = baseline_coords_A
    evaluation_count = 1

    if initial_values_deg.size > 0:
        bounds = [(-180.0, 180.0)] * int(initial_values_deg.size)

        def _objective(values_deg: np.ndarray) -> float:
            nonlocal evaluation_count
            evaluation_count += 1
            try:
                coords_A = engine.build_coords(np.asarray(values_deg, dtype=float))
                energy = float(engine.soft_energy_kj_mol(coords_A))
                if not np.isfinite(energy):
                    return float(1.0e12)
                return energy
            except Exception:
                return float(1.0e12)

        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=seed,
            maxiter=max(1, int(spec.symmetry_maxiter)),
            popsize=max(1, int(spec.symmetry_popsize)),
            polish=bool(spec.symmetry_polish),
            updating="deferred",
            workers=1,
        )
        best_values_deg = np.asarray(result.x, dtype=float)
        best_coords_A = engine.build_coords(best_values_deg)

    final_mol = update_rdkit_coords(
        engine.mol,
        (best_coords_A / 10.0) * unit.nanometer,
    )
    final_soft_energy = engine.soft_energy_kj_mol(best_coords_A)
    final_full_energy = (
        final_soft_energy
        if bool(spec.skip_full_stage)
        else engine.full_energy_kj_mol(best_coords_A)
    )

    baseline_hb, baseline_dmin, baseline_class_min, baseline_stacking = _ordering_diagnostics(
        baseline_mol,
        selector,
        spec,
    )
    final_hb, final_dmin, final_class_min, final_stacking = _ordering_diagnostics(
        final_mol,
        selector,
        spec,
    )
    baseline_symmetry = selector_screw_symmetry_rmsd_from_mol(baseline_mol)
    final_symmetry = selector_screw_symmetry_rmsd_from_mol(final_mol)

    selected_pose = engine.measure_dihedrals_deg(best_coords_A)
    summary: Dict[str, object] = {
        "enabled": True,
        "strategy": "symmetry_coupled",
        "objective": _ordering_objective_label(spec),
        "search_objective": "soft_single_point_energy_kj_mol",
        "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
        "hbond_connectivity_policy_applied": str(
            resolve_hbond_connectivity_policy(
                final_mol,
                selector,
                requested_policy=spec.hbond_connectivity_policy,
                requested_sites=engine.sites,
            )
        ),
        "stage1_nonbonded_mode": engine.soft_nonbonded_mode,
        "soft_exception_summary": dict(engine.soft_exception_summary),
        "stage2_nonbonded_mode": engine.full_nonbonded_mode,
        "full_stage_skipped": bool(spec.skip_full_stage),
        "final_stage_nonbonded_mode": (
            engine.soft_nonbonded_mode
            if bool(spec.skip_full_stage)
            else engine.full_nonbonded_mode
        ),
        "repeat_residues": 1,
        "helix_repeat_residues": (
            int(mol.GetIntProp("_poly_csp_helix_repeat_residues"))
            if mol.HasProp("_poly_csp_helix_repeat_residues")
            else None
        ),
        "candidate_count": (
            int(spec.symmetry_popsize) * max(1, len(engine.active_dihedrals))
        ),
        "evaluation_count": evaluation_count,
        "initialization_mode": "reference_residue_projection",
        "site_sweep_count": 0,
        "seed": seed,
        "active_symmetry_dof": len(engine.active_dihedrals),
        "symmetry_mode": "exact_screw_projection",
        "baseline_energy_kj_mol": float(baseline_full_energy),
        "baseline_stage1_energies_kj_mol": [float(baseline_soft_energy)],
        "baseline_stage2_energies_kj_mol": (
            [float(baseline_soft_energy)]
            if bool(spec.skip_full_stage)
            else [float(baseline_full_energy)]
        ),
        "baseline_hbond_like_fraction": baseline_hb.like_fraction,
        "baseline_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "baseline_hbond_donor_count": baseline_hb.donor_count,
        "baseline_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "baseline_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "baseline_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "baseline_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "baseline_min_heavy_distance_A": float(baseline_dmin),
        "baseline_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in baseline_class_min.items()
        },
        "baseline_selector_aromatic_stacking_A": dict(baseline_stacking),
        "baseline_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "initial_energy_kj_mol": float(baseline_full_energy),
        "initial_hbond_like_fraction": baseline_hb.like_fraction,
        "initial_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "initial_hbond_donor_count": baseline_hb.donor_count,
        "initial_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "initial_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "initial_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "initial_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "initial_selector_aromatic_stacking_A": dict(baseline_stacking),
        "initial_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "final_energy_kj_mol": float(final_full_energy),
        "final_score": -float(final_full_energy),
        "final_stage1_energies_kj_mol": [float(final_soft_energy)],
        "final_stage2_energies_kj_mol": (
            [float(final_soft_energy)]
            if bool(spec.skip_full_stage)
            else [float(final_full_energy)]
        ),
        "final_hbond_like_fraction": final_hb.like_fraction,
        "final_hbond_geometric_fraction": final_hb.geometric_fraction,
        "final_hbond_donor_count": final_hb.donor_count,
        "final_hbond_like_satisfied_donors": final_hb.like_satisfied_donors,
        "final_hbond_geometric_satisfied_donors": final_hb.geometric_satisfied_donors,
        "final_hbond_like_donor_occupancy_fraction": (
            final_hb.like_donor_occupancy_fraction
        ),
        "final_hbond_geometric_donor_occupancy_fraction": (
            final_hb.geometric_donor_occupancy_fraction
        ),
        "final_min_heavy_distance_A": float(final_dmin),
        "final_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in final_class_min.items()
        },
        "final_selector_aromatic_stacking_A": dict(final_stacking),
        "final_selector_symmetry_rmsd_A": float(final_symmetry),
        "soft_search_energy_kj_mol": float(final_soft_energy),
        "initial_pose_by_site": _symmetry_pose_summary(initial_pose),
        "selected_pose_by_site": _symmetry_pose_summary(selected_pose),
        "optimized_dihedrals_by_site": {
            site: {name: float(value) for name, value in sorted(values.items())}
            for site, values in selected_pose.items()
        },
    }
    return final_mol, summary


def optimize_symmetry_network_ordering(
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
        symmetry_rmsd = selector_screw_symmetry_rmsd_from_mol(mol)
        applied_hbond_connectivity_policy = resolve_hbond_connectivity_policy(
            mol,
            selector,
            requested_policy=spec.hbond_connectivity_policy,
            requested_sites=sites,
        )
        return Chem.Mol(mol), {
            "enabled": False,
            "strategy": "symmetry_network",
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
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "baseline_selector_aromatic_stacking_A": dict(stacking),
            "baseline_selector_symmetry_rmsd_A": float(symmetry_rmsd),
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
            "initial_selector_symmetry_rmsd_A": float(symmetry_rmsd),
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
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "final_selector_aromatic_stacking_A": dict(stacking),
            "final_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "initial_pose_by_site": {},
            "selected_pose_by_site": {},
        }

    engine = _build_engine(
        mol,
        selector=selector,
        sites=[str(site) for site in sites],
        dp=dp,
        spec=spec,
        grid=grid,
        include_anchor_dihedrals=True,
        runtime_params=runtime_params,
        mixing_rules_cfg=mixing_rules_cfg,
    )
    initial_pose = engine.initial_dihedrals_deg()
    initial_values_deg = np.asarray(
        [initial_pose[term.site][term.name] for term in engine.active_dihedrals],
        dtype=float,
    )
    baseline_coords_A = engine.build_coords(initial_values_deg)
    baseline_soft_energy = float(engine.soft_energy_kj_mol(baseline_coords_A))
    baseline_full_energy = (
        baseline_soft_energy
        if engine.full_context is None
        else float(engine.full_energy_kj_mol(baseline_coords_A))
    )
    eval_cache: dict[tuple[float, ...], _CandidateEval] = {}
    baseline_candidate = _evaluate_network_candidate(
        engine,
        selector=selector,
        values_deg=initial_values_deg,
        baseline_soft_energy=baseline_soft_energy,
        baseline_full_energy=baseline_full_energy,
        cache=eval_cache,
    )
    best_candidate = baseline_candidate
    evaluation_count = 1
    result = None

    if initial_values_deg.size > 0:
        bounds = [(-180.0, 180.0)] * int(initial_values_deg.size)
        init_population = (
            _initial_population_from_rotamers(
                engine,
                initial_values_deg=initial_values_deg,
                popsize=int(spec.symmetry_popsize),
                jitter_deg=float(spec.symmetry_init_jitter_deg),
                seed=seed,
            )
            if bool(spec.symmetry_init_from_rotamer_grid)
            else None
        )

        def _objective(values_deg: np.ndarray) -> float:
            nonlocal evaluation_count
            evaluation_count += 1
            try:
                candidate = _evaluate_network_candidate(
                    engine,
                    selector=selector,
                    values_deg=np.asarray(values_deg, dtype=float),
                    baseline_soft_energy=baseline_soft_energy,
                    baseline_full_energy=baseline_full_energy,
                    cache=eval_cache,
                )
                if not np.isfinite(candidate.score):
                    return float(1.0e12)
                return float(candidate.score)
            except Exception:
                return float(1.0e12)

        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=seed,
            maxiter=max(1, int(spec.symmetry_maxiter)),
            popsize=max(1, int(spec.symmetry_popsize)),
            polish=bool(spec.symmetry_polish),
            updating="deferred",
            workers=1,
            init=("latinhypercube" if init_population is None else init_population),
        )

        if bool(spec.symmetry_network_rerank_population):
            candidate_pool: list[_CandidateEval] = [baseline_candidate]
            population = getattr(result, "population", None)
            if population is not None:
                for values in np.asarray(population, dtype=float):
                    candidate_pool.append(
                        _evaluate_network_candidate(
                            engine,
                            selector=selector,
                            values_deg=np.asarray(values, dtype=float),
                            baseline_soft_energy=baseline_soft_energy,
                            baseline_full_energy=baseline_full_energy,
                            cache=eval_cache,
                        )
                    )
            candidate_pool.append(
                _evaluate_network_candidate(
                    engine,
                    selector=selector,
                    values_deg=np.asarray(result.x, dtype=float),
                    baseline_soft_energy=baseline_soft_energy,
                    baseline_full_energy=baseline_full_energy,
                    cache=eval_cache,
                )
            )
            best_candidate = _rank_network_candidates(candidate_pool)
        else:
            best_candidate = _evaluate_network_candidate(
                engine,
                selector=selector,
                values_deg=np.asarray(result.x, dtype=float),
                baseline_soft_energy=baseline_soft_energy,
                baseline_full_energy=baseline_full_energy,
                cache=eval_cache,
            )

    best_coords_A = np.asarray(best_candidate.coords_A, dtype=float)
    final_mol = update_rdkit_coords(engine.mol, (best_coords_A / 10.0) * unit.nanometer)
    final_soft_energy = float(engine.soft_energy_kj_mol(best_coords_A))
    final_full_energy = (
        final_soft_energy
        if engine.full_context is None
        else float(engine.full_energy_kj_mol(best_coords_A))
    )

    baseline_mol = update_rdkit_coords(engine.mol, (baseline_coords_A / 10.0) * unit.nanometer)
    baseline_hb, baseline_dmin, baseline_class_min, baseline_stacking = _ordering_diagnostics(
        baseline_mol,
        selector,
        spec,
    )
    final_hb, final_dmin, final_class_min, final_stacking = _ordering_diagnostics(
        final_mol,
        selector,
        spec,
    )
    baseline_symmetry = selector_screw_symmetry_rmsd_from_mol(baseline_mol)
    final_symmetry = selector_screw_symmetry_rmsd_from_mol(final_mol)

    selected_pose = engine.measure_dihedrals_deg(best_coords_A)
    active_anchor_dihedrals = sorted(
        {
            term.name
            for term in engine.active_dihedrals
            if term.name in selector.anchor_dihedrals
        }
    )
    applied_hbond_connectivity_policy = resolve_hbond_connectivity_policy(
        engine.mol,
        selector,
        requested_policy=spec.hbond_connectivity_policy,
        requested_sites=engine.sites,
    )
    summary: Dict[str, object] = {
        "enabled": True,
        "strategy": "symmetry_network",
        "objective": _ordering_objective_label(spec),
        "search_objective": "network_first_symmetry_score",
        "stage1_nonbonded_mode": engine.soft_nonbonded_mode,
        "soft_exception_summary": dict(engine.soft_exception_summary),
        "stage2_nonbonded_mode": engine.full_nonbonded_mode,
        "full_stage_skipped": bool(spec.skip_full_stage),
        "final_stage_nonbonded_mode": (
            engine.soft_nonbonded_mode
            if bool(spec.skip_full_stage)
            else engine.full_nonbonded_mode
        ),
        "repeat_residues": 1,
        "helix_repeat_residues": (
            int(mol.GetIntProp("_poly_csp_helix_repeat_residues"))
            if mol.HasProp("_poly_csp_helix_repeat_residues")
            else None
        ),
        "candidate_count": int(spec.symmetry_popsize) * max(1, len(engine.active_dihedrals)),
        "evaluation_count": evaluation_count,
        "initialization_mode": (
            "rotamer_grid_seeded_de"
            if bool(spec.symmetry_init_from_rotamer_grid)
            else "reference_residue_projection"
        ),
        "site_sweep_count": 0,
        "seed": seed,
        "active_symmetry_dof": len(engine.active_dihedrals),
        "active_anchor_dihedral_names": active_anchor_dihedrals,
        "symmetry_mode": "exact_screw_projection",
        "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
        "hbond_connectivity_policy_applied": str(applied_hbond_connectivity_policy),
        "network_score_weights": {
            "geom_occ": float(spec.symmetry_network_weight_geom_occ),
            "like_occ": float(spec.symmetry_network_weight_like_occ),
            "geom_frac": float(spec.symmetry_network_weight_geom_frac),
            "family_min_geom": float(spec.symmetry_network_weight_family_min_geom),
            "family_min_like": float(spec.symmetry_network_weight_family_min_like),
            "soft_energy": float(spec.symmetry_network_weight_soft_energy),
            "full_energy": float(spec.symmetry_network_weight_full_energy),
        },
        "network_score_clash_settings": {
            "min_heavy_distance_A": float(spec.symmetry_network_min_heavy_distance_A),
            "clash_penalty": float(spec.symmetry_network_clash_penalty),
            "energy_clip": float(spec.symmetry_network_energy_clip),
            "use_full_energy_in_search": bool(spec.symmetry_network_use_full_energy_in_search),
            "rerank_population": bool(spec.symmetry_network_rerank_population),
        },
        "final_selection_method": (
            "population_rerank" if bool(spec.symmetry_network_rerank_population) else "de_best"
        ),
        "baseline_energy_kj_mol": float(baseline_full_energy),
        "baseline_stage1_energies_kj_mol": [float(baseline_soft_energy)],
        "baseline_stage2_energies_kj_mol": (
            [float(baseline_soft_energy)]
            if engine.full_context is None
            else [float(baseline_full_energy)]
        ),
        "baseline_hbond_like_fraction": baseline_hb.like_fraction,
        "baseline_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "baseline_hbond_donor_count": baseline_hb.donor_count,
        "baseline_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "baseline_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "baseline_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "baseline_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "baseline_hbond_family_metrics": _hbond_family_metrics_summary(
            baseline_candidate.hbond_family_metrics
        ),
        "baseline_hbond_family_min_like_fraction": float(
            baseline_candidate.hbond_family_min_like_fraction
        ),
        "baseline_hbond_family_min_geometric_fraction": float(
            baseline_candidate.hbond_family_min_geometric_fraction
        ),
        "baseline_min_heavy_distance_A": float(baseline_dmin),
        "baseline_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in baseline_class_min.items()
        },
        "baseline_selector_aromatic_stacking_A": dict(baseline_stacking),
        "baseline_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "initial_energy_kj_mol": float(baseline_full_energy),
        "initial_hbond_like_fraction": baseline_hb.like_fraction,
        "initial_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "initial_hbond_donor_count": baseline_hb.donor_count,
        "initial_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "initial_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "initial_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "initial_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "initial_hbond_family_metrics": _hbond_family_metrics_summary(
            baseline_candidate.hbond_family_metrics
        ),
        "initial_hbond_family_min_like_fraction": float(
            baseline_candidate.hbond_family_min_like_fraction
        ),
        "initial_hbond_family_min_geometric_fraction": float(
            baseline_candidate.hbond_family_min_geometric_fraction
        ),
        "initial_selector_aromatic_stacking_A": dict(baseline_stacking),
        "initial_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "final_energy_kj_mol": float(final_full_energy),
        "final_score": -float(best_candidate.score),
        "final_stage1_energies_kj_mol": [float(final_soft_energy)],
        "final_stage2_energies_kj_mol": (
            [float(final_soft_energy)]
            if engine.full_context is None
            else [float(final_full_energy)]
        ),
        "final_hbond_like_fraction": final_hb.like_fraction,
        "final_hbond_geometric_fraction": final_hb.geometric_fraction,
        "final_hbond_donor_count": final_hb.donor_count,
        "final_hbond_like_satisfied_donors": final_hb.like_satisfied_donors,
        "final_hbond_geometric_satisfied_donors": final_hb.geometric_satisfied_donors,
        "final_hbond_like_donor_occupancy_fraction": (
            final_hb.like_donor_occupancy_fraction
        ),
        "final_hbond_geometric_donor_occupancy_fraction": (
            final_hb.geometric_donor_occupancy_fraction
        ),
        "final_hbond_family_metrics": _hbond_family_metrics_summary(
            best_candidate.hbond_family_metrics
        ),
        "final_hbond_family_min_like_fraction": float(
            best_candidate.hbond_family_min_like_fraction
        ),
        "final_hbond_family_min_geometric_fraction": float(
            best_candidate.hbond_family_min_geometric_fraction
        ),
        "final_min_heavy_distance_A": float(final_dmin),
        "final_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in final_class_min.items()
        },
        "final_selector_aromatic_stacking_A": dict(final_stacking),
        "final_selector_symmetry_rmsd_A": float(final_symmetry),
        "soft_search_energy_kj_mol": float(best_candidate.soft_energy_kj_mol),
        "initial_pose_by_site": _symmetry_pose_summary(initial_pose),
        "selected_pose_by_site": _symmetry_pose_summary(selected_pose),
        "optimized_dihedrals_by_site": {
            site: {name: float(value) for name, value in sorted(values.items())}
            for site, values in selected_pose.items()
        },
    }
    if result is not None:
        summary["de_result_fun"] = float(result.fun)
    return final_mol, summary
