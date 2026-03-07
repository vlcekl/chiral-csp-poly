from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import openmm as mm
from openmm import unit

from poly_csp.forcefield.anneal import run_heat_cool_cycle, run_temperature_ramp
from poly_csp.forcefield.runtime_params import RuntimeParams, load_runtime_params
from poly_csp.forcefield.system_builder import create_system
from poly_csp.forcefield.restraints import (
    add_dihedral_restraints,
    add_hbond_distance_restraints,
    add_positional_restraints,
)
from poly_csp.structure.dihedrals import measure_dihedral_rad
from poly_csp.topology.atom_mapping import selector_instance_maps
from poly_csp.topology.selectors import SelectorTemplate


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


def _manifest_source(atom: Chem.Atom) -> str:
    if atom.HasProp("_poly_csp_manifest_source"):
        return str(atom.GetProp("_poly_csp_manifest_source"))
    return "backbone"


def _backbone_heavy_indices(mol: Chem.Mol) -> list[int]:
    return [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and _manifest_source(atom) == "backbone"
    ]


def _backbone_all_indices(mol: Chem.Mol) -> list[int]:
    return [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if _manifest_source(atom) == "backbone"
    ]


def _selector_all_indices(mol: Chem.Mol) -> set[int]:
    return {
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if _manifest_source(atom) == "selector"
    }


def _connector_all_indices(mol: Chem.Mol) -> set[int]:
    return {
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if _manifest_source(atom) == "connector"
    }


def _selector_dihedral_targets(
    mol: Chem.Mol,
    selector: SelectorTemplate | None,
) -> list[tuple[int, int, int, int, float]]:
    if selector is None or mol.GetNumConformers() == 0:
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
    if selector is None or mol.GetNumConformers() == 0:
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
            donors.append((inst, int(atom.GetIdx())))
        if local in selector.acceptors:
            acceptors.append((inst, int(atom.GetIdx())))

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


def _update_rdkit_coords(mol: Chem.Mol, positions_nm: unit.Quantity) -> Chem.Mol:
    xyz_A = positions_nm.value_in_unit(unit.nanometer) * 10.0
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(xyz_A):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def _require_forcefield_molecule(mol: Chem.Mol) -> None:
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Relaxation requires a forcefield-domain molecule from build_forcefield_molecule()."
        )


def _prepare_system_for_relaxation(
    system: mm.System,
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None,
    reference_positions_nm: unit.Quantity,
) -> None:
    backbone_heavy = _backbone_heavy_indices(mol)
    if spec.freeze_backbone:
        for idx in backbone_heavy:
            system.setParticleMass(int(idx), 0.0)

    add_positional_restraints(
        system=system,
        atom_indices=backbone_heavy,
        reference_positions_nm=reference_positions_nm,
        k_kj_per_mol_nm2=float(spec.positional_k),
    )
    add_dihedral_restraints(
        system=system,
        dihedrals=_selector_dihedral_targets(mol, selector),
        k_kj_per_mol=float(spec.dihedral_k),
    )
    add_hbond_distance_restraints(
        system=system,
        pairs=_hbond_pairs(mol, selector),
        k_kj_per_mol_nm2=float(spec.hbond_k),
    )


def _new_context(
    system: mm.System,
    positions_nm: unit.Quantity,
) -> tuple[mm.Context, mm.LangevinIntegrator]:
    integrator = mm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    context = mm.Context(system, integrator)
    context.setPositions(positions_nm)
    return context, integrator


def _potential_energy_kj_mol(context: mm.Context) -> float:
    state = context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _run_minimization_schedule(
    context: mm.Context,
    spec: RelaxSpec,
    factors: np.ndarray,
) -> list[float]:
    energies: list[float] = []
    for factor in factors:
        context.setParameter("k_pos", float(spec.positional_k) * float(factor))
        context.setParameter("k_tors", float(spec.dihedral_k) * float(factor))
        context.setParameter("k_hb", float(spec.hbond_k) * float(factor))
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        energies.append(_potential_energy_kj_mol(context))
    return energies


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


def run_staged_relaxation(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
    *,
    runtime_params: RuntimeParams | None = None,
    work_dir: str | Path | None = None,
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

    reference_positions_nm = _positions_nm_from_mol(mol)

    soft_result = create_system(
        mol,
        glycam_params=runtime.glycam,
        selector_params_by_name=runtime.selector_params_by_name,
        connector_params_by_key=runtime.connector_params_by_key,
        parameter_provenance=runtime.source_manifest,
        nonbonded_mode="soft",
        mixing_rules_cfg=mixing_rules_cfg,
    )
    _prepare_system_for_relaxation(
        soft_result.system,
        mol,
        spec,
        selector,
        reference_positions_nm,
    )
    soft_context, soft_integrator = _new_context(soft_result.system, soft_result.positions_nm)
    stage_factors = np.linspace(1.0, 0.15, max(1, int(spec.n_stages)))
    soft_energies = _run_minimization_schedule(soft_context, spec, stage_factors)
    soft_state = soft_context.getState(getPositions=True)
    stage1_positions = soft_state.getPositions(asNumpy=True)
    del soft_context, soft_integrator

    full_result = create_system(
        mol,
        glycam_params=runtime.glycam,
        selector_params_by_name=runtime.selector_params_by_name,
        connector_params_by_key=runtime.connector_params_by_key,
        parameter_provenance=runtime.source_manifest,
        nonbonded_mode="full",
        mixing_rules_cfg=mixing_rules_cfg,
    )
    _prepare_system_for_relaxation(
        full_result.system,
        mol,
        spec,
        selector,
        reference_positions_nm,
    )
    full_context, full_integrator = _new_context(full_result.system, stage1_positions)
    final_factor = float(stage_factors[-1])
    full_context.setParameter("k_pos", float(spec.positional_k) * final_factor)
    full_context.setParameter("k_tors", float(spec.dihedral_k) * final_factor)
    full_context.setParameter("k_hb", float(spec.hbond_k) * final_factor)

    stage2_energies = []
    mm.LocalEnergyMinimizer.minimize(
        full_context,
        tolerance=10.0,
        maxIterations=int(spec.max_iterations),
    )
    stage2_energies.append(_potential_energy_kj_mol(full_context))

    if spec.anneal_enabled and int(spec.anneal_steps) > 0:
        _apply_stage2_anneal(full_context, full_integrator, spec)
        mm.LocalEnergyMinimizer.minimize(
            full_context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        stage2_energies.append(_potential_energy_kj_mol(full_context))

    final_state = full_context.getState(getPositions=True, getEnergy=True)
    final_positions = final_state.getPositions(asNumpy=True)

    final_xyz_A = np.asarray(final_positions.value_in_unit(unit.nanometer)) * 10.0
    span = final_xyz_A.max(axis=0) - final_xyz_A.min(axis=0)

    backbone_heavy = _backbone_heavy_indices(mol)
    backbone_all = _backbone_all_indices(mol)
    selector_indices = _selector_all_indices(mol)
    connector_indices = _connector_all_indices(mol)
    backbone_drift = 0.0
    if spec.freeze_backbone and backbone_heavy:
        init_backbone_xyz = (
            np.asarray(reference_positions_nm.value_in_unit(unit.nanometer))[backbone_heavy]
            * 10.0
        )
        final_backbone_xyz = final_xyz_A[backbone_heavy]
        backbone_drift = float(np.max(np.abs(final_backbone_xyz - init_backbone_xyz)))

    out = _update_rdkit_coords(mol, final_positions)
    summary: Dict[str, object] = {
        "enabled": True,
        "protocol": "two_stage_runtime",
        "stage1_nonbonded_mode": soft_result.nonbonded_mode,
        "stage1_energies_kj_mol": soft_energies,
        "stage2_nonbonded_mode": full_result.nonbonded_mode,
        "stage2_energies_kj_mol": stage2_energies,
        "anneal_enabled": bool(spec.anneal_enabled),
        "anneal_cool_down": bool(spec.anneal_cool_down),
        "freeze_backbone": bool(spec.freeze_backbone),
        "n_backbone_atoms": len(backbone_all),
        "n_backbone_heavy_frozen": len(backbone_heavy) if spec.freeze_backbone else 0,
        "n_selector_atoms": len(selector_indices),
        "n_connector_atoms": len(connector_indices),
        "component_counts": dict(full_result.component_counts),
        "soft_exception_summary": dict(soft_result.exception_summary),
        "full_exception_summary": dict(full_result.exception_summary),
        "source_manifest": dict(full_result.source_manifest),
        "span_A": [float(value) for value in span.tolist()],
        "backbone_drift_A": float(backbone_drift),
    }
    return out, summary
