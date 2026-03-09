from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from rdkit import Chem

import openmm as mm
from openmm import unit

from poly_csp.forcefield.anneal import run_heat_cool_cycle, run_temperature_ramp
from poly_csp.forcefield.minimization import (
    HELIX_CORE_BACKBONE_ATOM_NAMES,
    PreparedRuntimeOptimizationBundle,
    RuntimeRestraintSpec,
    TwoStageMinimizationProtocol,
    backbone_all_indices,
    backbone_heavy_indices,
    connector_all_indices,
    new_context,
    potential_energy_kj_mol,
    prepare_runtime_optimization_bundle,
    run_prepared_runtime_optimization,
    set_optional_parameter,
    selector_all_indices,
    update_rdkit_coords,
)
from poly_csp.forcefield.runtime_params import RuntimeParams, load_runtime_params
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class RelaxSpec:
    enabled: bool
    positional_k: float
    dihedral_k: float
    hbond_k: float
    soft_n_stages: int = 3
    soft_max_iterations: int = 200
    full_max_iterations: int = 200
    final_restraint_factor: float = 0.15
    freeze_backbone: bool = True
    anneal_enabled: bool = False
    t_start_K: float = 50.0
    t_end_K: float = 350.0
    anneal_steps: int = 2000
    anneal_cool_down: bool = True


def _require_forcefield_molecule(mol: Chem.Mol) -> None:
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Relaxation requires a forcefield-domain molecule from build_forcefield_molecule()."
        )


def _apply_stage2_anneal(
    context: mm.Context,
    integrator: mm.LangevinIntegrator,
    spec: RelaxSpec,
) -> None:
    if not spec.anneal_enabled or int(spec.anneal_steps) <= 0:
        return
    if spec.anneal_cool_down:
        run_heat_cool_cycle(
            context=context,
            integrator=integrator,
            t_start_K=float(spec.t_start_K),
            t_peak_K=float(spec.t_end_K),
            n_steps=int(spec.anneal_steps),
            n_segments=10,
        )
        return
    run_temperature_ramp(
        context=context,
        integrator=integrator,
        t_start_K=float(spec.t_start_K),
        t_end_K=float(spec.t_end_K),
        n_steps=int(spec.anneal_steps),
        n_segments=10,
    )


def _relaxation_restraint_spec(spec: RelaxSpec) -> RuntimeRestraintSpec:
    return RuntimeRestraintSpec(
        positional_k=float(spec.positional_k),
        dihedral_k=float(spec.dihedral_k),
        hbond_k=float(spec.hbond_k),
        freeze_backbone=bool(spec.freeze_backbone),
    )


def _relaxation_protocol(spec: RelaxSpec) -> TwoStageMinimizationProtocol:
    return TwoStageMinimizationProtocol(
        soft_n_stages=int(spec.soft_n_stages),
        soft_max_iterations=int(spec.soft_max_iterations),
        full_max_iterations=int(spec.full_max_iterations),
        final_restraint_factor=float(spec.final_restraint_factor),
    )


def _prepare_relaxation_bundle(
    mol: Chem.Mol,
    *,
    runtime: RuntimeParams,
    selector: SelectorTemplate | None,
    spec: RelaxSpec,
    mixing_rules_cfg: Mapping[str, object] | None,
    soft_repulsion_k_kj_per_mol_nm2: float,
    soft_repulsion_cutoff_nm: float,
) -> PreparedRuntimeOptimizationBundle:
    return prepare_runtime_optimization_bundle(
        mol,
        runtime_params=runtime,
        selector=selector,
        mixing_rules_cfg=mixing_rules_cfg,
        restraint_spec=_relaxation_restraint_spec(spec),
        protocol=_relaxation_protocol(spec),
        soft_repulsion_k_kj_per_mol_nm2=float(soft_repulsion_k_kj_per_mol_nm2),
        soft_repulsion_cutoff_nm=float(soft_repulsion_cutoff_nm),
    )


def run_staged_relaxation(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
    *,
    runtime_params: RuntimeParams | None = None,
    work_dir: str | Path | None = None,
    soft_repulsion_k_kj_per_mol_nm2: float = 800.0,
    soft_repulsion_cutoff_nm: float = 0.6,
    mixing_rules_cfg: Mapping[str, object] | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    """Run the canonical two-stage runtime relaxation on a forcefield-domain molecule."""
    if not spec.enabled:
        return Chem.Mol(mol), {"enabled": False}

    _require_forcefield_molecule(mol)

    runtime = runtime_params
    if runtime is None:
        runtime = load_runtime_params(
            mol,
            selector_template=selector,
            work_dir=None if work_dir is None else Path(work_dir),
        )
    bundle = _prepare_relaxation_bundle(
        mol,
        runtime=runtime,
        selector=selector,
        spec=spec,
        mixing_rules_cfg=mixing_rules_cfg,
        soft_repulsion_k_kj_per_mol_nm2=float(soft_repulsion_k_kj_per_mol_nm2),
        soft_repulsion_cutoff_nm=float(soft_repulsion_cutoff_nm),
    )
    minimization = run_prepared_runtime_optimization(bundle)
    stage2_energies = list(minimization.stage2_energies_kj_mol)
    anneal_final_energy: float | None = None

    if spec.anneal_enabled and int(spec.anneal_steps) > 0:
        full_context, full_integrator = new_context(
            bundle.full.system,
            minimization.final_positions_nm,
        )
        final_factor = float(bundle.protocol.final_restraint_factor)
        set_optional_parameter(
            full_context,
            "k_pos",
            float(bundle.restraint_spec.positional_k) * final_factor,
        )
        set_optional_parameter(
            full_context,
            "k_tors",
            float(bundle.restraint_spec.dihedral_k) * final_factor,
        )
        set_optional_parameter(
            full_context,
            "k_hb",
            float(bundle.restraint_spec.hbond_k) * final_factor,
        )
        _apply_stage2_anneal(full_context, full_integrator, spec)
        mm.LocalEnergyMinimizer.minimize(
            full_context,
            tolerance=10.0,
            maxIterations=int(spec.full_max_iterations),
        )
        anneal_final_energy = potential_energy_kj_mol(full_context)
        final_positions = full_context.getState(getPositions=True).getPositions(asNumpy=True)
        del full_context, full_integrator
    else:
        final_positions = minimization.final_positions_nm

    final_xyz_A = np.asarray(final_positions.value_in_unit(unit.nanometer)) * 10.0
    span = final_xyz_A.max(axis=0) - final_xyz_A.min(axis=0)

    helix_core_heavy = backbone_heavy_indices(mol)
    backbone_all = backbone_all_indices(mol)
    selector_indices = selector_all_indices(mol)
    connector_indices = connector_all_indices(mol)
    backbone_drift = 0.0
    if spec.freeze_backbone and helix_core_heavy:
        init_backbone_xyz = (
            np.asarray(bundle.reference_positions_nm.value_in_unit(unit.nanometer))[helix_core_heavy]
            * 10.0
        )
        final_backbone_xyz = final_xyz_A[helix_core_heavy]
        backbone_drift = float(np.max(np.abs(final_backbone_xyz - init_backbone_xyz)))

    out = update_rdkit_coords(mol, final_positions)
    summary: Dict[str, object] = {
        "enabled": True,
        "protocol": "two_stage_runtime",
        "protocol_summary": asdict(bundle.protocol),
        "restraint_summary": asdict(bundle.restraint_spec),
        "stage1_nonbonded_mode": bundle.soft.nonbonded_mode,
        "stage1_energies_kj_mol": list(minimization.stage1_energies_kj_mol),
        "stage2_nonbonded_mode": bundle.full.nonbonded_mode,
        "stage2_energies_kj_mol": stage2_energies,
        "final_energy_kj_mol": (
            float(anneal_final_energy)
            if anneal_final_energy is not None
            else float(minimization.stage2_energies_kj_mol[-1])
        ),
        "anneal_enabled": bool(spec.anneal_enabled),
        "anneal_cool_down": bool(spec.anneal_cool_down),
        "anneal_summary": {
            "enabled": bool(spec.anneal_enabled),
            "steps": int(spec.anneal_steps),
            "cool_down": bool(spec.anneal_cool_down),
            "t_start_K": float(spec.t_start_K),
            "t_end_K": float(spec.t_end_K),
            "final_energy_kj_mol": anneal_final_energy,
        },
        "freeze_backbone": bool(spec.freeze_backbone),
        "n_backbone_atoms": len(backbone_all),
        "helix_core_atom_names": sorted(HELIX_CORE_BACKBONE_ATOM_NAMES),
        "n_backbone_heavy_frozen": len(helix_core_heavy) if spec.freeze_backbone else 0,
        "n_helix_core_heavy_frozen": len(helix_core_heavy) if spec.freeze_backbone else 0,
        "n_selector_atoms": len(selector_indices),
        "n_connector_atoms": len(connector_indices),
        "component_counts": dict(bundle.full.component_counts),
        "soft_exception_summary": dict(bundle.soft.exception_summary),
        "full_exception_summary": dict(bundle.full.exception_summary),
        "soft_force_inventory": asdict(bundle.soft.force_inventory),
        "full_force_inventory": asdict(bundle.full.force_inventory),
        "source_manifest": dict(bundle.full.source_manifest),
        "span_A": [float(value) for value in span.tolist()],
        "backbone_drift_A": float(backbone_drift),
    }
    return out, summary
