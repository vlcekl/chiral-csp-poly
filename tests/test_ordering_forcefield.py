from __future__ import annotations

import math

from poly_csp.ordering.scoring import selector_aromatic_ring_planarity
from poly_csp.ordering.optimize import OrderingSpec, optimize_selector_ordering
from poly_csp.structure.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.structure.selector_library.tmb import make_tmb_template
from tests.support import build_forcefield_mol, make_fake_runtime_params


def _ordering_spec(*, repeat_residues: int = 1, max_candidates: int = 8) -> OrderingSpec:
    return OrderingSpec(
        enabled=True,
        repeat_residues=repeat_residues,
        max_candidates=max_candidates,
        positional_k=1000.0,
        soft_n_stages=1,
        soft_max_iterations=5,
        full_max_iterations=5,
    )


def test_optimize_selector_ordering_returns_forcefield_summary() -> None:
    selector = make_35_dmpc_template()
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
    assert "final_class_min_distance_A" in summary
    assert "selected_pose_by_site" in summary
    assert "C6" in summary["selected_pose_by_site"]


def test_optimize_selector_ordering_supports_cellulose_runtime_systems() -> None:
    selector = make_tmb_template()
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
    selector = make_35_dmpc_template()
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


def test_ordering_seeded_determinism_and_metadata() -> None:
    selector = make_35_dmpc_template()
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


def test_ordering_repeat_unit_summary_uses_repeat_positions() -> None:
    selector = make_35_dmpc_template()
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


def test_optimize_selector_ordering_requires_forcefield_molecule() -> None:
    selector = make_35_dmpc_template()
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
    selector = make_35_dmpc_template()
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
