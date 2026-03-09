from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import openmm as mm
from openmm import unit

from poly_csp.forcefield.restraints import (
    add_dihedral_restraints,
    add_hbond_distance_restraints,
    add_positional_restraints,
)
from poly_csp.forcefield.runtime_params import RuntimeParams
from poly_csp.forcefield.system_builder import SystemBuildResult, create_system
from poly_csp.structure.dihedrals import measure_dihedral_rad
from poly_csp.topology.atom_mapping import selector_instance_maps
from poly_csp.topology.selectors import SelectorTemplate


HELIX_CORE_BACKBONE_ATOM_NAMES = frozenset({"C1", "C2", "C3", "C4", "C5", "O4", "O5"})


@dataclass(frozen=True)
class RuntimeRestraintSpec:
    positional_k: float = 0.0
    dihedral_k: float = 0.0
    hbond_k: float = 0.0
    freeze_backbone: bool = True


@dataclass(frozen=True)
class TwoStageMinimizationProtocol:
    soft_n_stages: int = 3
    soft_max_iterations: int = 200
    full_max_iterations: int = 200
    final_restraint_factor: float = 0.15


@dataclass(frozen=True)
class PreparedRuntimeOptimizationBundle:
    soft: SystemBuildResult
    full: SystemBuildResult
    reference_positions_nm: unit.Quantity
    restraint_spec: RuntimeRestraintSpec
    protocol: TwoStageMinimizationProtocol


@dataclass(frozen=True)
class TwoStageMinimizationResult:
    stage1_energies_kj_mol: tuple[float, ...]
    stage2_energies_kj_mol: tuple[float, ...]
    stage1_positions_nm: unit.Quantity
    final_positions_nm: unit.Quantity


def manifest_source(atom: Chem.Atom) -> str:
    if atom.HasProp("_poly_csp_manifest_source"):
        return str(atom.GetProp("_poly_csp_manifest_source"))
    return "backbone"


def backbone_heavy_indices(mol: Chem.Mol) -> list[int]:
    return helix_core_backbone_heavy_indices(mol)


def helix_core_backbone_heavy_indices(mol: Chem.Mol) -> list[int]:
    return [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if (
            atom.GetAtomicNum() > 1
            and manifest_source(atom) == "backbone"
            and atom.HasProp("_poly_csp_atom_name")
            and atom.GetProp("_poly_csp_atom_name") in HELIX_CORE_BACKBONE_ATOM_NAMES
        )
    ]


def backbone_all_indices(mol: Chem.Mol) -> list[int]:
    return [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if manifest_source(atom) == "backbone"
    ]


def selector_all_indices(mol: Chem.Mol) -> set[int]:
    return {
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if manifest_source(atom) == "selector"
    }


def connector_all_indices(mol: Chem.Mol) -> set[int]:
    return {
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if manifest_source(atom) == "connector"
    }


def selector_dihedral_targets(
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


def hbond_pairs(
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


def positions_nm_from_mol(mol: Chem.Mol) -> unit.Quantity:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before OpenMM minimization.")
    xyz_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    return (xyz_A / 10.0) * unit.nanometer


def update_rdkit_coords(mol: Chem.Mol, positions_nm: unit.Quantity) -> Chem.Mol:
    xyz_A = np.asarray(positions_nm.value_in_unit(unit.nanometer), dtype=float) * 10.0
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for atom_idx, (x, y, z) in enumerate(xyz_A):
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def prepare_system_for_minimization(
    system: mm.System,
    mol: Chem.Mol,
    restraint_spec: RuntimeRestraintSpec,
    selector: SelectorTemplate | None,
    reference_positions_nm: unit.Quantity,
) -> None:
    backbone_heavy = backbone_heavy_indices(mol)
    if restraint_spec.freeze_backbone:
        for idx in backbone_heavy:
            system.setParticleMass(int(idx), 0.0)

    if float(restraint_spec.positional_k) > 0.0 and backbone_heavy:
        add_positional_restraints(
            system=system,
            atom_indices=backbone_heavy,
            reference_positions_nm=reference_positions_nm,
            k_kj_per_mol_nm2=float(restraint_spec.positional_k),
        )
    if selector is not None and float(restraint_spec.dihedral_k) > 0.0:
        add_dihedral_restraints(
            system=system,
            dihedrals=selector_dihedral_targets(mol, selector),
            k_kj_per_mol=float(restraint_spec.dihedral_k),
        )
    if selector is not None and float(restraint_spec.hbond_k) > 0.0:
        add_hbond_distance_restraints(
            system=system,
            pairs=hbond_pairs(mol, selector),
            k_kj_per_mol_nm2=float(restraint_spec.hbond_k),
        )


def prepare_runtime_optimization_bundle(
    mol: Chem.Mol,
    *,
    runtime_params: RuntimeParams,
    selector: SelectorTemplate | None,
    mixing_rules_cfg: Mapping[str, object] | None,
    restraint_spec: RuntimeRestraintSpec,
    protocol: TwoStageMinimizationProtocol,
    soft_repulsion_k_kj_per_mol_nm2: float = 800.0,
    soft_repulsion_cutoff_nm: float = 0.6,
) -> PreparedRuntimeOptimizationBundle:
    reference_positions_nm = positions_nm_from_mol(mol)
    soft = create_system(
        mol,
        glycam_params=runtime_params.glycam,
        selector_params_by_name=runtime_params.selector_params_by_name,
        connector_params_by_key=runtime_params.connector_params_by_key,
        parameter_provenance=runtime_params.source_manifest,
        nonbonded_mode="soft",
        repulsion_k_kj_per_mol_nm2=float(soft_repulsion_k_kj_per_mol_nm2),
        repulsion_cutoff_nm=float(soft_repulsion_cutoff_nm),
        mixing_rules_cfg=mixing_rules_cfg,
    )
    prepare_system_for_minimization(
        system=soft.system,
        mol=mol,
        restraint_spec=restraint_spec,
        selector=selector,
        reference_positions_nm=reference_positions_nm,
    )
    full = create_system(
        mol,
        glycam_params=runtime_params.glycam,
        selector_params_by_name=runtime_params.selector_params_by_name,
        connector_params_by_key=runtime_params.connector_params_by_key,
        parameter_provenance=runtime_params.source_manifest,
        nonbonded_mode="full",
        mixing_rules_cfg=mixing_rules_cfg,
    )
    prepare_system_for_minimization(
        system=full.system,
        mol=mol,
        restraint_spec=restraint_spec,
        selector=selector,
        reference_positions_nm=reference_positions_nm,
    )
    return PreparedRuntimeOptimizationBundle(
        soft=soft,
        full=full,
        reference_positions_nm=reference_positions_nm,
        restraint_spec=restraint_spec,
        protocol=protocol,
    )


def run_prepared_runtime_optimization(
    bundle: PreparedRuntimeOptimizationBundle,
    *,
    initial_positions_nm: unit.Quantity | None = None,
) -> TwoStageMinimizationResult:
    return run_two_stage_minimization(
        soft_system=bundle.soft.system,
        full_system=bundle.full.system,
        initial_positions_nm=(
            bundle.soft.positions_nm
            if initial_positions_nm is None
            else initial_positions_nm
        ),
        restraint_spec=bundle.restraint_spec,
        protocol=bundle.protocol,
    )


def new_context(
    system: mm.System,
    positions_nm: unit.Quantity,
) -> tuple[mm.Context, mm.LangevinIntegrator]:
    integrator = mm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    context: mm.Context | None = None
    for platform_name, properties in (("CPU", {"Threads": "1"}), ("Reference", {})):
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            if properties:
                context = mm.Context(system, integrator, platform, properties)
            else:
                context = mm.Context(system, integrator, platform)
            break
        except Exception:
            context = None
    if context is None:
        context = mm.Context(system, integrator)
    context.setPositions(positions_nm)
    return context, integrator


def potential_energy_kj_mol(context: mm.Context) -> float:
    state = context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def set_optional_parameter(context: mm.Context, name: str, value: float) -> None:
    try:
        context.setParameter(name, float(value))
    except mm.OpenMMException as exc:
        if "invalid parameter name" not in str(exc):
            raise


def _run_minimization_schedule(
    context: mm.Context,
    *,
    positional_k: float,
    dihedral_k: float,
    hbond_k: float,
    max_iterations: int,
    factors: np.ndarray,
) -> list[float]:
    energies: list[float] = []
    for factor in factors:
        set_optional_parameter(context, "k_pos", float(positional_k) * float(factor))
        set_optional_parameter(context, "k_tors", float(dihedral_k) * float(factor))
        set_optional_parameter(context, "k_hb", float(hbond_k) * float(factor))
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(max_iterations),
        )
        energies.append(potential_energy_kj_mol(context))
    return energies


def run_two_stage_minimization(
    *,
    soft_system: mm.System,
    full_system: mm.System,
    initial_positions_nm: unit.Quantity,
    restraint_spec: RuntimeRestraintSpec,
    protocol: TwoStageMinimizationProtocol,
) -> TwoStageMinimizationResult:
    n_stages = max(1, int(protocol.soft_n_stages))
    stage_factors = np.linspace(1.0, float(protocol.final_restraint_factor), n_stages)

    soft_context, soft_integrator = new_context(soft_system, initial_positions_nm)
    stage1_energies = _run_minimization_schedule(
        soft_context,
        positional_k=float(restraint_spec.positional_k),
        dihedral_k=float(restraint_spec.dihedral_k),
        hbond_k=float(restraint_spec.hbond_k),
        max_iterations=int(protocol.soft_max_iterations),
        factors=stage_factors,
    )
    stage1_positions = soft_context.getState(getPositions=True).getPositions(asNumpy=True)
    del soft_context, soft_integrator

    full_context, full_integrator = new_context(full_system, stage1_positions)
    final_factor = float(stage_factors[-1])
    set_optional_parameter(
        full_context,
        "k_pos",
        float(restraint_spec.positional_k) * final_factor,
    )
    set_optional_parameter(
        full_context,
        "k_tors",
        float(restraint_spec.dihedral_k) * final_factor,
    )
    set_optional_parameter(
        full_context,
        "k_hb",
        float(restraint_spec.hbond_k) * final_factor,
    )
    mm.LocalEnergyMinimizer.minimize(
        full_context,
        tolerance=10.0,
        maxIterations=int(protocol.full_max_iterations),
    )
    stage2_energies = [potential_energy_kj_mol(full_context)]
    final_positions = full_context.getState(getPositions=True).getPositions(asNumpy=True)
    del full_context, full_integrator

    return TwoStageMinimizationResult(
        stage1_energies_kj_mol=tuple(float(x) for x in stage1_energies),
        stage2_energies_kj_mol=tuple(float(x) for x in stage2_energies),
        stage1_positions_nm=stage1_positions,
        final_positions_nm=final_positions,
    )
