from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import openmm as mm
from openmm import unit

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.minimization import (
    ExplicitPositionalRestraintGroup,
    HELIX_CORE_BACKBONE_ATOM_NAMES,
    PreparedRuntimeOptimizationBundle,
    RuntimeRestraintSpec,
    TwoStageMinimizationProtocol,
    TwoStageMinimizationResult,
    helix_core_backbone_heavy_indices,
    positions_nm_from_mol,
    prepare_system_for_minimization,
)
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.relaxation import RelaxSpec, run_staged_relaxation
from poly_csp.forcefield.system_builder import (
    ForceInventorySummary,
    SystemBuildResult,
)
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.terminals import apply_terminal_mode


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-4.71238898038469,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=4.933333333333334,
        handedness="left",
    )


def _forcefield_selector_mol():
    selector = SelectorRegistry.get("35dmpc")
    template = make_glucose_template("amylose", monomer_representation="anhydro")
    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    topology = apply_terminal_mode(
        mol=topology,
        mode="open",
        caps={},
        representation="anhydro",
    )
    structure = build_backbone_structure(topology, _helix()).mol
    structure = attach_selector(
        mol_polymer=structure,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    return build_forcefield_molecule(structure).mol, selector


def _fake_system_result(mol, nonbonded_mode: str) -> SystemBuildResult:
    system = mm.System()
    for atom in mol.GetAtoms():
        mass = 1.0 if atom.GetAtomicNum() == 1 else 12.0
        system.addParticle(mass)

    bond_force = mm.HarmonicBondForce()
    for bond in mol.GetBonds():
        bond_force.addBond(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            0.15,
            50.0,
        )
    system.addForce(bond_force)

    if nonbonded_mode == "soft":
        repulsive = mm.CustomNonbondedForce("0")
        repulsive.addPerParticleParameter("sigma")
        for _ in mol.GetAtoms():
            repulsive.addParticle([0.2])
        system.addForce(repulsive)
        exception_summary = {"mode": "soft", "num_exclusions": 0}
        force_inventory = ForceInventorySummary(
            forces=("HarmonicBondForce", "CustomNonbondedForce"),
            counts={"CustomNonbondedForce": 1, "HarmonicBondForce": 1},
        )
    else:
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        for _ in mol.GetAtoms():
            nonbonded.addParticle(0.0, 0.2, 0.0)
        system.addForce(nonbonded)
        exception_summary = {"exceptions_seen": 0, "exceptions_patched": 0}
        force_inventory = ForceInventorySummary(
            forces=("HarmonicBondForce", "NonbondedForce"),
            counts={"HarmonicBondForce": 1, "NonbondedForce": 1},
        )

    xyz = np.asarray(
        build_forcefield_molecule(mol).mol.GetConformer(0).GetPositions(),
        dtype=float,
    )
    positions_nm = (xyz / 10.0) * unit.nanometer
    return SystemBuildResult(
        system=system,
        positions_nm=positions_nm,
        excluded_pairs=set(),
        nonbonded_mode=nonbonded_mode,
        topology_manifest=(),
        component_counts={"backbone": 1, "selector": 1, "connector": 1},
        force_inventory=force_inventory,
        exception_summary=exception_summary,
        source_manifest={"fake": True},
    )


def test_run_staged_relaxation_uses_shared_runtime_bundle(monkeypatch) -> None:
    mol, selector = _forcefield_selector_mol()
    runtime = SimpleNamespace(
        glycam=None,
        selector_params_by_name={},
        connector_params_by_key={},
        source_manifest={"runtime": {"cache": {"kind": "test"}}},
    )
    calls: dict[str, object] = {}
    soft = _fake_system_result(mol, nonbonded_mode="soft")
    full = _fake_system_result(mol, nonbonded_mode="full")
    bundle = PreparedRuntimeOptimizationBundle(
        soft=soft,
        full=full,
        reference_positions_nm=soft.positions_nm,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=10.0,
            dihedral_k=0.0,
            hbond_k=0.0,
            freeze_backbone=False,
        ),
        protocol=TwoStageMinimizationProtocol(
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=7,
            final_restraint_factor=0.2,
        ),
    )

    def fake_prepare_runtime_optimization_bundle(*args, **kwargs):
        calls["prepare"] = {"args": args, "kwargs": kwargs}
        return bundle

    def fake_run_prepared_runtime_optimization(prepared, *, initial_positions_nm=None):
        calls["run"] = {
            "prepared": prepared,
            "initial_positions_nm": initial_positions_nm,
        }
        return TwoStageMinimizationResult(
            stage1_energies_kj_mol=(12.0,),
            stage2_energies_kj_mol=(8.5,),
            stage1_positions_nm=prepared.soft.positions_nm,
            final_positions_nm=prepared.full.positions_nm,
        )

    monkeypatch.setattr(
        "poly_csp.forcefield.relaxation.prepare_runtime_optimization_bundle",
        fake_prepare_runtime_optimization_bundle,
    )
    monkeypatch.setattr(
        "poly_csp.forcefield.relaxation.run_prepared_runtime_optimization",
        fake_run_prepared_runtime_optimization,
    )

    spec = RelaxSpec(
        enabled=True,
        positional_k=10.0,
        dihedral_k=0.0,
        hbond_k=0.0,
        soft_n_stages=1,
        soft_max_iterations=5,
        full_max_iterations=7,
        final_restraint_factor=0.2,
        freeze_backbone=False,
        anti_stacking_sigma_scale=1.4,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.19,
        hbond_neighbor_window=2,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="donor_heavy",
        anneal_enabled=False,
    )
    relaxed, summary = run_staged_relaxation(
        mol=mol,
        spec=spec,
        selector=selector,
        runtime_params=runtime,
    )

    assert calls["prepare"]["args"][0] is mol
    assert calls["prepare"]["kwargs"]["selector"] is selector
    assert calls["prepare"]["kwargs"]["runtime_params"] is runtime
    assert calls["prepare"]["kwargs"]["soft_exclude_14"] is True
    assert calls["prepare"]["kwargs"]["anti_stacking_sigma_scale"] == 1.4
    assert calls["prepare"]["kwargs"]["ideal_hbond_target_nm"] == 0.19
    assert calls["prepare"]["kwargs"]["hbond_neighbor_window"] == 2
    assert calls["prepare"]["kwargs"]["hbond_pairing_mode"] == "nearest_unique"
    assert calls["prepare"]["kwargs"]["hbond_restraint_atom_mode"] == "donor_heavy"
    assert calls["run"]["prepared"] is bundle
    assert calls["run"]["initial_positions_nm"] is None
    assert relaxed.GetNumAtoms() == mol.GetNumAtoms()
    assert relaxed.GetNumConformers() == 1
    assert summary["enabled"] is True
    assert summary["protocol"] == "two_stage_runtime"
    assert summary["protocol_summary"] == asdict(bundle.protocol)
    assert summary["restraint_summary"] == asdict(bundle.restraint_spec)
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] == "full"
    assert summary["full_stage_skipped"] is False
    assert summary["final_stage_nonbonded_mode"] == "full"
    assert summary["stage1_energies_kj_mol"] == [12.0]
    assert summary["stage2_energies_kj_mol"] == [8.5]
    assert summary["final_energy_kj_mol"] == 8.5
    assert summary["anneal_summary"]["final_energy_kj_mol"] is None
    assert summary["source_manifest"] == {"fake": True}
    assert summary["soft_force_inventory"]["counts"]["CustomNonbondedForce"] == 1
    assert summary["full_force_inventory"]["counts"]["NonbondedForce"] == 1


def test_run_staged_relaxation_skip_full_stage_updates_summary(monkeypatch) -> None:
    mol, selector = _forcefield_selector_mol()
    runtime = SimpleNamespace(
        glycam=None,
        selector_params_by_name={},
        connector_params_by_key={},
        source_manifest={"runtime": {"cache": {"kind": "test"}}},
    )
    soft = _fake_system_result(mol, nonbonded_mode="soft")
    full = _fake_system_result(mol, nonbonded_mode="full")
    bundle = PreparedRuntimeOptimizationBundle(
        soft=soft,
        full=full,
        reference_positions_nm=soft.positions_nm,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=10.0,
            dihedral_k=0.0,
            hbond_k=0.0,
            freeze_backbone=False,
        ),
        protocol=TwoStageMinimizationProtocol(
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=7,
            final_restraint_factor=0.2,
            skip_full_stage=True,
        ),
    )

    monkeypatch.setattr(
        "poly_csp.forcefield.relaxation.prepare_runtime_optimization_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        "poly_csp.forcefield.relaxation.run_prepared_runtime_optimization",
        lambda prepared, *, initial_positions_nm=None: TwoStageMinimizationResult(
            stage1_energies_kj_mol=(12.0,),
            stage2_energies_kj_mol=(12.0,),
            stage1_positions_nm=prepared.soft.positions_nm,
            final_positions_nm=prepared.soft.positions_nm,
            full_stage_skipped=True,
        ),
    )

    spec = RelaxSpec(
        enabled=True,
        positional_k=10.0,
        dihedral_k=0.0,
        hbond_k=0.0,
        soft_n_stages=1,
        soft_max_iterations=5,
        full_max_iterations=7,
        final_restraint_factor=0.2,
        freeze_backbone=False,
        skip_full_stage=True,
        anneal_enabled=False,
    )
    _, summary = run_staged_relaxation(
        mol=mol,
        spec=spec,
        selector=selector,
        runtime_params=runtime,
    )

    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] is None
    assert summary["full_stage_skipped"] is True
    assert summary["final_stage_nonbonded_mode"] == "soft"
    assert summary["stage1_energies_kj_mol"] == [12.0]
    assert summary["stage2_energies_kj_mol"] == [12.0]
    assert summary["final_energy_kj_mol"] == 12.0


def test_run_staged_relaxation_rejects_skip_full_stage_with_anneal() -> None:
    mol, selector = _forcefield_selector_mol()

    spec = RelaxSpec(
        enabled=True,
        positional_k=10.0,
        dihedral_k=0.0,
        hbond_k=0.0,
        skip_full_stage=True,
        anneal_enabled=True,
    )

    try:
        run_staged_relaxation(
            mol=mol,
            spec=spec,
            selector=selector,
            runtime_params=SimpleNamespace(
                glycam=None,
                selector_params_by_name={},
                connector_params_by_key={},
                source_manifest={"runtime": {"cache": {"kind": "test"}}},
            ),
        )
    except ValueError as exc:
        assert "skip_full_stage=True" in str(exc)
    else:
        raise AssertionError("Expected skip_full_stage+anneal to fail.")


def test_helix_core_backbone_heavy_indices_exclude_exocyclic_site_atoms() -> None:
    mol, _ = _forcefield_selector_mol()
    indices = helix_core_backbone_heavy_indices(mol)
    atom_names = {
        mol.GetAtomWithIdx(idx).GetProp("_poly_csp_atom_name")
        for idx in indices
    }

    assert HELIX_CORE_BACKBONE_ATOM_NAMES.issubset(atom_names)
    assert "C6" not in atom_names
    assert "O1" not in atom_names
    assert "O2" not in atom_names
    assert "O3" not in atom_names
    assert "O6" not in atom_names


def test_prepare_system_for_minimization_adds_explicit_reference_restraints() -> None:
    mol, _ = _forcefield_selector_mol()
    system = mm.System()
    for _ in mol.GetAtoms():
        system.addParticle(12.0)

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    group = ExplicitPositionalRestraintGroup(
        atom_indices=(0, 1),
        reference_positions_A=(
            tuple(float(v) for v in xyz[0]),
            tuple(float(v) for v in xyz[1]),
        ),
        k_kj_per_mol_nm2=123.0,
        parameter_name="k_pos_ref_test",
        label="test_group",
    )

    prepare_system_for_minimization(
        system=system,
        mol=mol,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=0.0,
            dihedral_k=0.0,
            hbond_k=0.0,
            freeze_backbone=False,
        ),
        selector=None,
        reference_positions_nm=positions_nm_from_mol(mol),
        extra_positional_restraints=(group,),
    )

    external_forces = [
        system.getForce(force_idx)
        for force_idx in range(system.getNumForces())
        if isinstance(system.getForce(force_idx), mm.CustomExternalForce)
    ]
    assert len(external_forces) == 1
    force = external_forces[0]
    assert force.getGlobalParameterName(0) == "k_pos_ref_test"
    assert force.getNumParticles() == 2
