from __future__ import annotations

import math
from types import SimpleNamespace

from poly_csp.config.schema import HelixSpec, SoftSelectorHbondBiasOptions
from poly_csp.forcefield.minimization import (
    TwoStageMinimizationProtocol,
    TwoStageMinimizationResult,
    positions_nm_from_mol,
)
from poly_csp.ordering.hbonds import HbondMetrics
from poly_csp.ordering.scoring import (
    selector_aromatic_ring_planarity,
    selector_aromatic_stacking_metrics,
)
from poly_csp.ordering.optimize import OrderingSpec, optimize_selector_ordering
from poly_csp.topology.selectors import SelectorRegistry
from tests.support import build_forcefield_mol, make_fake_runtime_params


def _ordering_spec(
    *,
    repeat_residues: int | None = None,
    max_candidates: int = 8,
    strategy: str = "greedy",
    symmetry_maxiter: int = 60,
    symmetry_popsize: int = 12,
) -> OrderingSpec:
    return OrderingSpec(
        enabled=True,
        strategy=strategy,  # type: ignore[arg-type]
        repeat_residues=repeat_residues,
        max_candidates=max_candidates,
        positional_k=1000.0,
        soft_n_stages=1,
        soft_max_iterations=5,
        full_max_iterations=5,
        symmetry_maxiter=symmetry_maxiter,
        symmetry_popsize=symmetry_popsize,
    )


def test_optimize_selector_ordering_returns_forcefield_summary() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=3, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")

    out, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        spec=_ordering_spec(),
        runtime_params=runtime_params,
    )

    assert out.HasProp("_poly_csp_manifest_schema_version")
    assert out.GetNumAtoms() == mol.GetNumAtoms()
    assert out.GetNumConformers() == 1
    assert summary["enabled"] is True
    assert summary["objective"] == "negative_stage2_energy_kj_mol"
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] == "full"
    assert summary["final_score"] == -summary["final_energy_kj_mol"]
    assert "final_hbond_geometric_fraction" in summary
    assert "final_hbond_like_donor_occupancy_fraction" in summary
    assert "baseline_hbond_like_donor_occupancy_fraction" in summary
    assert "final_class_min_distance_A" in summary
    assert "final_selector_aromatic_stacking_A" in summary
    assert "selected_pose_by_site" in summary
    assert "C6" in summary["selected_pose_by_site"]


def test_optimize_selector_ordering_skip_full_stage_updates_summary() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=3, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")

    out, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        spec=OrderingSpec(
            enabled=True,
            max_candidates=4,
            positional_k=1000.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            skip_full_stage=True,
        ),
        runtime_params=runtime_params,
    )

    assert out.HasProp("_poly_csp_manifest_schema_version")
    assert summary["objective"] == "negative_stage1_energy_kj_mol"
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] is None
    assert summary["full_stage_skipped"] is True
    assert summary["final_stage_nonbonded_mode"] == "soft"
    assert summary["final_score"] == -summary["final_energy_kj_mol"]


def test_optimize_selector_ordering_forwards_hbond_bias_to_runtime_bundle(
    monkeypatch,
) -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")
    calls: dict[str, object] = {}
    positions_nm = positions_nm_from_mol(mol)
    hbond_metrics = HbondMetrics(
        like_satisfied_pairs=0,
        geometric_satisfied_pairs=0,
        total_pairs=0,
        donor_count=0,
        like_satisfied_donors=0,
        geometric_satisfied_donors=0,
        like_fraction=0.0,
        geometric_fraction=0.0,
        like_donor_occupancy_fraction=0.0,
        geometric_donor_occupancy_fraction=0.0,
        mean_like_distance_A=0.0,
        mean_geometric_distance_A=0.0,
    )
    bundle = SimpleNamespace(
        soft=SimpleNamespace(
            nonbonded_mode="soft",
            positions_nm=positions_nm,
            exception_summary={},
        ),
        full=SimpleNamespace(nonbonded_mode="full", positions_nm=positions_nm),
        protocol=TwoStageMinimizationProtocol(
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            final_restraint_factor=0.15,
        ),
    )

    def fake_prepare_runtime_optimization_bundle(*args, **kwargs):
        calls["prepare_args"] = args
        calls["prepare_kwargs"] = kwargs
        return bundle

    def fake_run_prepared_runtime_optimization(prepared, *, initial_positions_nm=None):
        return TwoStageMinimizationResult(
            stage1_energies_kj_mol=(7.0,),
            stage2_energies_kj_mol=(5.0,),
            stage1_positions_nm=initial_positions_nm,
            final_positions_nm=initial_positions_nm,
            full_stage_skipped=False,
        )

    monkeypatch.setattr(
        "poly_csp.ordering.optimize.prepare_runtime_optimization_bundle",
        fake_prepare_runtime_optimization_bundle,
    )
    monkeypatch.setattr(
        "poly_csp.ordering.optimize.run_prepared_runtime_optimization",
        fake_run_prepared_runtime_optimization,
    )
    monkeypatch.setattr(
        "poly_csp.ordering.optimize._ordering_diagnostics",
        lambda *args, **kwargs: (
            hbond_metrics,
            2.0,
            {"selector_selector": 3.0},
            {"min_centroid_distance_A": 4.2},
        ),
    )

    _, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=1,
        spec=OrderingSpec(
            enabled=True,
            max_candidates=1,
            positional_k=1000.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            soft_repulsion_k_kj_per_mol_nm2=910.0,
            soft_repulsion_cutoff_nm=0.64,
            hbond_k=25.0,
            ideal_hbond_target_nm=0.19,
            hbond_pairing_mode="nearest_unique",
            hbond_restraint_atom_mode="donor_heavy",
            soft_selector_hbond_bias=SoftSelectorHbondBiasOptions(enabled=True),
        ),
        runtime_params=runtime_params,
    )

    assert calls["prepare_args"][0].GetNumAtoms() == mol.GetNumAtoms()
    assert calls["prepare_kwargs"]["selector"] is selector
    assert calls["prepare_kwargs"]["restraint_spec"].hbond_k == 25.0
    assert calls["prepare_kwargs"]["soft_repulsion_k_kj_per_mol_nm2"] == 910.0
    assert calls["prepare_kwargs"]["soft_repulsion_cutoff_nm"] == 0.64
    assert calls["prepare_kwargs"]["ideal_hbond_target_nm"] == 0.19
    assert calls["prepare_kwargs"]["hbond_pairing_mode"] == "nearest_unique"
    assert calls["prepare_kwargs"]["hbond_restraint_atom_mode"] == "donor_heavy"
    assert calls["prepare_kwargs"]["soft_selector_hbond_bias"].enabled is True
    assert summary["final_score"] == -summary["final_energy_kj_mol"]


def test_optimize_selector_ordering_rebuilds_runtime_bundle_when_hbond_bias_enabled(
    monkeypatch,
) -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")
    prepare_calls: list[int] = []
    positions_nm = positions_nm_from_mol(mol)
    hbond_metrics = HbondMetrics(
        like_satisfied_pairs=0,
        geometric_satisfied_pairs=0,
        total_pairs=0,
        donor_count=0,
        like_satisfied_donors=0,
        geometric_satisfied_donors=0,
        like_fraction=0.0,
        geometric_fraction=0.0,
        like_donor_occupancy_fraction=0.0,
        geometric_donor_occupancy_fraction=0.0,
        mean_like_distance_A=0.0,
        mean_geometric_distance_A=0.0,
    )
    bundle = SimpleNamespace(
        soft=SimpleNamespace(
            nonbonded_mode="soft",
            positions_nm=positions_nm,
            exception_summary={},
        ),
        full=SimpleNamespace(nonbonded_mode="full", positions_nm=positions_nm),
        protocol=TwoStageMinimizationProtocol(
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            final_restraint_factor=0.15,
        ),
    )

    def fake_prepare_runtime_optimization_bundle(*args, **kwargs):
        prepare_calls.append(1)
        return bundle

    def fake_run_prepared_runtime_optimization(prepared, *, initial_positions_nm=None):
        return TwoStageMinimizationResult(
            stage1_energies_kj_mol=(7.0,),
            stage2_energies_kj_mol=(5.0,),
            stage1_positions_nm=initial_positions_nm,
            final_positions_nm=initial_positions_nm,
            full_stage_skipped=False,
        )

    monkeypatch.setattr(
        "poly_csp.ordering.optimize.prepare_runtime_optimization_bundle",
        fake_prepare_runtime_optimization_bundle,
    )
    monkeypatch.setattr(
        "poly_csp.ordering.optimize.run_prepared_runtime_optimization",
        fake_run_prepared_runtime_optimization,
    )
    monkeypatch.setattr(
        "poly_csp.ordering.optimize._ordering_diagnostics",
        lambda *args, **kwargs: (
            hbond_metrics,
            2.0,
            {"selector_selector": 3.0},
            {"min_centroid_distance_A": 4.2},
        ),
    )

    optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=1,
        spec=OrderingSpec(
            enabled=True,
            max_candidates=2,
            positional_k=1000.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            hbond_k=25.0,
            max_site_sweeps=1,
            randomize_initial_assignment=False,
            randomize_site_order=False,
            randomize_residue_order=False,
            randomize_pose_order=False,
        ),
        runtime_params=runtime_params,
    )

    assert len(prepare_calls) > 1


def test_optimize_selector_ordering_reuses_shared_runtime_bundle_with_soft_selector_hbond_bias(
    monkeypatch,
) -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")
    prepare_calls: list[int] = []
    positions_nm = positions_nm_from_mol(mol)
    hbond_metrics = HbondMetrics(
        like_satisfied_pairs=0,
        geometric_satisfied_pairs=0,
        total_pairs=0,
        donor_count=0,
        like_satisfied_donors=0,
        geometric_satisfied_donors=0,
        like_fraction=0.0,
        geometric_fraction=0.0,
        like_donor_occupancy_fraction=0.0,
        geometric_donor_occupancy_fraction=0.0,
        mean_like_distance_A=0.0,
        mean_geometric_distance_A=0.0,
    )
    bundle = SimpleNamespace(
        soft=SimpleNamespace(
            nonbonded_mode="soft",
            positions_nm=positions_nm,
            exception_summary={
                "soft_selector_hbond_bias": {"enabled": True, "pair_count": 2}
            },
        ),
        full=SimpleNamespace(nonbonded_mode="full", positions_nm=positions_nm),
        protocol=TwoStageMinimizationProtocol(
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            final_restraint_factor=0.15,
        ),
    )

    def fake_prepare_runtime_optimization_bundle(*args, **kwargs):
        prepare_calls.append(1)
        return bundle

    def fake_run_prepared_runtime_optimization(prepared, *, initial_positions_nm=None):
        return TwoStageMinimizationResult(
            stage1_energies_kj_mol=(7.0,),
            stage2_energies_kj_mol=(5.0,),
            stage1_positions_nm=initial_positions_nm,
            final_positions_nm=initial_positions_nm,
            full_stage_skipped=False,
        )

    monkeypatch.setattr(
        "poly_csp.ordering.optimize.prepare_runtime_optimization_bundle",
        fake_prepare_runtime_optimization_bundle,
    )
    monkeypatch.setattr(
        "poly_csp.ordering.optimize.run_prepared_runtime_optimization",
        fake_run_prepared_runtime_optimization,
    )
    monkeypatch.setattr(
        "poly_csp.ordering.optimize._ordering_diagnostics",
        lambda *args, **kwargs: (
            hbond_metrics,
            2.0,
            {"selector_selector": 3.0},
            {"min_centroid_distance_A": 4.2},
        ),
    )

    _, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=1,
        spec=OrderingSpec(
            enabled=True,
            max_candidates=2,
            positional_k=1000.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            hbond_k=0.0,
            soft_selector_hbond_bias=SoftSelectorHbondBiasOptions(enabled=True),
            max_site_sweeps=1,
            randomize_initial_assignment=False,
            randomize_site_order=False,
            randomize_residue_order=False,
            randomize_pose_order=False,
        ),
        runtime_params=runtime_params,
    )

    assert len(prepare_calls) == 1
    assert summary["soft_exception_summary"]["soft_selector_hbond_bias"]["enabled"] is True


def test_optimize_selector_ordering_supports_cellulose_runtime_systems() -> None:
    selector = SelectorRegistry.get("tmb")
    mol = build_forcefield_mol(polymer="cellulose", dp=2, selector=selector, site="C3")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C3")

    out, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C3"],
        dp=2,
        spec=_ordering_spec(max_candidates=4),
        runtime_params=runtime_params,
    )

    assert out.HasProp("_poly_csp_manifest_schema_version")
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] == "full"
    assert summary["final_energy_kj_mol"] is not None


def test_optimize_selector_ordering_supports_periodic_runtime_systems() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")

    out, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=4,
        spec=_ordering_spec(repeat_residues=4, max_candidates=4),
        runtime_params=runtime_params,
    )

    assert out.HasProp("_poly_csp_manifest_schema_version")
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] == "full"
    assert summary["final_energy_kj_mol"] is not None


def test_optimize_selector_ordering_supports_symmetry_coupled_strategy() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=3, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")

    out, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        spec=_ordering_spec(
            strategy="symmetry_coupled",
            symmetry_maxiter=1,
            symmetry_popsize=2,
        ),
        runtime_params=runtime_params,
        seed=17,
    )

    assert out.HasProp("_poly_csp_manifest_schema_version")
    assert summary["strategy"] == "symmetry_coupled"
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] == "full"
    assert summary["final_energy_kj_mol"] is not None
    assert summary["final_selector_symmetry_rmsd_A"] < 1e-6
    assert summary["active_symmetry_dof"] >= 1
    assert "optimized_dihedrals_by_site" in summary
    assert "C6" in summary["selected_pose_by_site"]


def test_optimize_selector_ordering_supports_symmetry_network_strategy() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=3, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")

    out, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        spec=OrderingSpec(
            enabled=True,
            strategy="symmetry_network",
            positional_k=1000.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            symmetry_maxiter=1,
            symmetry_popsize=2,
            symmetry_polish=False,
            symmetry_network_use_full_energy_in_search=False,
            symmetry_network_rerank_population=True,
        ),
        runtime_params=runtime_params,
        seed=23,
    )

    assert out.HasProp("_poly_csp_manifest_schema_version")
    assert summary["strategy"] == "symmetry_network"
    assert summary["search_objective"] == "staged_network_first_symmetry_score"
    assert summary["hbond_connectivity_policy_applied"] == "custom_v1"
    assert "c6_forward_neighbor" in summary["final_hbond_family_metrics"]
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] == "full"
    assert summary["final_energy_kj_mol"] is not None
    assert summary["final_selector_symmetry_rmsd_A"] < 1e-6
    assert summary["active_anchor_dihedral_names"] == ["tau_attach"]
    assert summary["network_capture_enabled"] is True
    assert summary["network_capture_applied"] is True
    assert summary["backbone_refinement_applied"] is True
    assert summary["backbone_refinement_skipped_reason"] is None
    assert summary["active_backbone_dihedral_names"] == ["bb_c6_omega"]
    assert "bb_c6_omega" in summary["backbone_refinement_initial_backbone_dihedrals_deg"]
    assert "bb_c6_omega" in summary["backbone_refinement_final_backbone_dihedrals_deg"]
    assert "tau_attach" in summary["selected_pose_by_site"]["C6"]["0"]
    assert summary["network_score_weights"]["geom_occ"] == 100.0
    assert summary["network_score_weights"]["family_min_geom"] == 150.0


def test_symmetry_network_open_chain_glycosidic_refinement_activates_for_zipper_sites() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=3, selector=selector, site="C2")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C2")

    _, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C2"],
        dp=3,
        spec=OrderingSpec(
            enabled=True,
            strategy="symmetry_network",
            positional_k=1000.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            symmetry_maxiter=1,
            symmetry_popsize=2,
            symmetry_backbone_maxiter=1,
            symmetry_backbone_popsize=2,
            symmetry_polish=False,
            symmetry_network_use_full_energy_in_search=False,
            symmetry_network_rerank_population=True,
        ),
        runtime_params=runtime_params,
        seed=19,
    )

    assert summary["strategy"] == "symmetry_network"
    assert summary["backbone_refinement_applied"] is True
    assert "bb_phi" in summary["active_backbone_dihedral_names"]
    assert "bb_psi" in summary["active_backbone_dihedral_names"]
    assert "bb_phi" in summary["backbone_refinement_initial_backbone_dihedrals_deg"]
    assert "bb_psi" in summary["backbone_refinement_initial_backbone_dihedrals_deg"]


def test_symmetry_network_periodic_glycosidic_refinement_activates_for_zipper_sites() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C2",
        end_mode="periodic",
    )
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C2")

    _, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C2"],
        dp=4,
        spec=OrderingSpec(
            enabled=True,
            strategy="symmetry_network",
            positional_k=1000.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            symmetry_maxiter=1,
            symmetry_popsize=2,
            symmetry_backbone_maxiter=1,
            symmetry_backbone_popsize=2,
            symmetry_polish=False,
            symmetry_network_use_full_energy_in_search=False,
            symmetry_network_rerank_population=True,
        ),
        runtime_params=runtime_params,
        seed=29,
    )

    assert summary["strategy"] == "symmetry_network"
    assert summary["backbone_refinement_applied"] is True
    assert summary["backbone_refinement_skipped_reason"] is None
    assert "bb_phi" in summary["active_backbone_dihedral_names"]
    assert "bb_psi" in summary["active_backbone_dihedral_names"]
    assert "bb_phi" in summary["backbone_refinement_initial_backbone_dihedrals_deg"]
    assert "bb_psi" in summary["backbone_refinement_initial_backbone_dihedrals_deg"]


def test_ordering_seeded_determinism_and_metadata() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=3, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")
    spec = _ordering_spec(max_candidates=4)

    _, summary1 = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        spec=spec,
        seed=42,
        runtime_params=runtime_params,
    )
    _, summary2 = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        spec=spec,
        seed=42,
        runtime_params=runtime_params,
    )
    _, summary3 = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        spec=spec,
        seed=99,
        runtime_params=runtime_params,
    )

    assert math.isclose(summary1["final_score"], summary2["final_score"], abs_tol=1e-2)
    assert math.isclose(
        summary1["final_energy_kj_mol"],
        summary2["final_energy_kj_mol"],
        abs_tol=1e-2,
    )
    assert summary1["seed"] == summary2["seed"] == 42
    assert summary3["seed"] == 99
    assert summary1["initialization_mode"] == "seeded_random_assignment"
    assert "C6" in summary1["initial_pose_by_site"]


def test_ordering_repeat_unit_summary_uses_repeat_positions() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=4, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")

    _, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=4,
        spec=_ordering_spec(repeat_residues=2, max_candidates=4),
        runtime_params=runtime_params,
    )

    assert summary["repeat_residues"] == 2
    assert set(summary["selected_pose_by_site"]["C6"]) == {"0", "1"}


def test_ordering_repeat_unit_defaults_to_helix_repeat_metadata() -> None:
    selector = SelectorRegistry.get("tmb")
    cellulose_helix = HelixSpec(
        name="cellulose_CSP_3_2_derivatized",
        repeat_residues=3,
        repeat_turns=2,
        axial_repeat_A=16.2,
        handedness="left",
    )
    mol = build_forcefield_mol(
        polymer="cellulose",
        dp=3,
        selector=selector,
        site="C3",
        helix=cellulose_helix,
    )
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C3")

    _, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C3"],
        dp=3,
        spec=_ordering_spec(max_candidates=4),
        runtime_params=runtime_params,
    )

    assert summary["repeat_residues"] == 3
    assert set(summary["selected_pose_by_site"]["C3"]) == {"0", "1", "2"}


def test_optimize_selector_ordering_requires_forcefield_molecule() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    mol.ClearProp("_poly_csp_manifest_schema_version")

    try:
        optimize_selector_ordering(
            mol=mol,
            selector=selector,
            sites=["C6"],
            dp=1,
            spec=_ordering_spec(max_candidates=4),
            runtime_params=make_fake_runtime_params(
                build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6"),
                selector=selector,
                site="C6",
            ),
        )
    except ValueError as exc:
        assert "forcefield-domain molecule" in str(exc)
    else:
        raise AssertionError("Expected non-forcefield ordering input to fail.")


def test_selector_aromatic_ring_planarity_detects_out_of_plane_distortion() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")

    baseline = selector_aromatic_ring_planarity(mol, selector.mol)
    assert baseline
    assert baseline["ring_count"] == 1

    distorted = type(mol)(mol)
    conf = distorted.GetConformer()
    aromatic_ring = next(
        ring
        for ring in selector.mol.GetRingInfo().AtomRings()
        if len(ring) == 6
        and all(selector.mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
    )
    local_idx = int(aromatic_ring[0])
    atom_idx = next(
        atom.GetIdx()
        for atom in distorted.GetAtoms()
        if atom.HasProp("_poly_csp_selector_instance")
        and int(atom.GetIntProp("_poly_csp_selector_instance")) == 1
        and int(atom.GetIntProp("_poly_csp_selector_local_idx")) == local_idx
    )
    pos = conf.GetAtomPosition(atom_idx)
    conf.SetAtomPosition(atom_idx, (float(pos.x), float(pos.y), float(pos.z) + 1.0))

    warped = selector_aromatic_ring_planarity(distorted, selector.mol)
    assert warped["max_out_of_plane_A"] > baseline["max_out_of_plane_A"] + 0.1
    assert warped["max_out_of_plane_A"] > 0.1


def test_selector_aromatic_stacking_metrics_detect_collapsed_selector_pair() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=2, selector=selector, site="C6")

    baseline = selector_aromatic_stacking_metrics(mol, selector.mol)
    assert baseline
    assert baseline["instance_count"] == 2
    assert baseline["instance_pair_count"] == 1
    assert baseline["ring_pair_count"] >= 1

    distorted = type(mol)(mol)
    conf = distorted.GetConformer()
    aromatic_ring = next(
        ring
        for ring in selector.mol.GetRingInfo().AtomRings()
        if all(selector.mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
    )
    instance_maps: dict[int, dict[int, int]] = {}
    for atom in distorted.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        instance_id = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local_idx = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        instance_maps.setdefault(instance_id, {})[local_idx] = int(atom.GetIdx())

    instance_ids = sorted(instance_maps)
    assert len(instance_ids) == 2
    first_instance, second_instance = instance_ids
    for local_idx in aromatic_ring:
        source_atom_idx = instance_maps[first_instance][int(local_idx)]
        target_atom_idx = instance_maps[second_instance][int(local_idx)]
        pos = conf.GetAtomPosition(source_atom_idx)
        conf.SetAtomPosition(
            target_atom_idx,
            (float(pos.x), float(pos.y), float(pos.z)),
        )

    collapsed = selector_aromatic_stacking_metrics(distorted, selector.mol)
    assert collapsed["min_centroid_distance_A"] < 0.1
    assert collapsed["ring_pairs_below_threshold"] >= 1
    assert (
        collapsed["min_centroid_distance_A"]
        < baseline["min_centroid_distance_A"]
    )
