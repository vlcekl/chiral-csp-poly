"""Tests for multi-start forcefield-aware selector ordering."""

from __future__ import annotations

import math

from poly_csp.ordering.multi_opt import MultiOptSpec, RankedResult, run_multi_start_optimization
from poly_csp.ordering.optimize import OrderingSpec
from poly_csp.topology.selectors import SelectorRegistry
from tests.support import build_forcefield_mol, make_fake_runtime_params


def _build_runtime_case(dp: int = 3):
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=dp, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")
    return mol, selector, runtime_params


def _ordering_spec(max_candidates: int = 8) -> OrderingSpec:
    return OrderingSpec(
        enabled=True,
        repeat_residues=1,
        max_candidates=max_candidates,
        positional_k=1000.0,
        soft_n_stages=1,
        soft_max_iterations=5,
        full_max_iterations=5,
    )


def test_multi_start_returns_ranked_results() -> None:
    mol, selector, runtime_params = _build_runtime_case(dp=3)
    ordering_spec = _ordering_spec(max_candidates=4)
    multi_spec = MultiOptSpec(enabled=True, n_starts=3, top_k=2, seed=42, n_workers=1)

    results = run_multi_start_optimization(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        ordering_spec=ordering_spec,
        multi_spec=multi_spec,
        runtime_params=runtime_params,
    )

    assert len(results) == 2
    assert all(isinstance(r, RankedResult) for r in results)
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].score >= results[1].score
    assert results[0].mol.GetNumAtoms() == mol.GetNumAtoms()
    assert results[0].summary["stage2_nonbonded_mode"] == "full"


def test_multi_start_reproducibility() -> None:
    mol, selector, runtime_params = _build_runtime_case(dp=3)
    ordering_spec = _ordering_spec(max_candidates=4)
    multi_spec = MultiOptSpec(enabled=True, n_starts=3, top_k=2, seed=123, n_workers=1)

    run1 = run_multi_start_optimization(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        ordering_spec=ordering_spec,
        multi_spec=multi_spec,
        runtime_params=runtime_params,
    )
    run2 = run_multi_start_optimization(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        ordering_spec=ordering_spec,
        multi_spec=multi_spec,
        runtime_params=runtime_params,
    )

    assert len(run1) == len(run2)
    assert sorted(result.seed_used for result in run1) == sorted(
        result.seed_used for result in run2
    )
    for r1, r2 in zip(
        sorted(run1, key=lambda result: result.seed_used),
        sorted(run2, key=lambda result: result.seed_used),
    ):
        assert math.isclose(r1.score, r2.score, abs_tol=1e-2)


def test_multi_start_no_seed_runs() -> None:
    mol, selector, runtime_params = _build_runtime_case(dp=2)
    ordering_spec = _ordering_spec(max_candidates=4)
    multi_spec = MultiOptSpec(enabled=True, n_starts=2, top_k=2, seed=None, n_workers=1)

    results = run_multi_start_optimization(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=2,
        ordering_spec=ordering_spec,
        multi_spec=multi_spec,
        runtime_params=runtime_params,
    )

    assert len(results) == 2
    assert results[0].rank == 1


def test_multi_start_falls_back_to_serial_when_process_pool_unavailable(
    monkeypatch,
) -> None:
    mol, selector, runtime_params = _build_runtime_case(dp=2)
    ordering_spec = _ordering_spec(max_candidates=4)
    multi_spec = MultiOptSpec(enabled=True, n_starts=2, top_k=2, seed=7, n_workers=2)

    class _FailingPool:
        def __init__(self, *args, **kwargs):
            raise PermissionError("blocked by sandbox")

    import poly_csp.ordering.multi_opt as multi_mod

    monkeypatch.setattr(multi_mod, "ProcessPoolExecutor", _FailingPool)

    results = run_multi_start_optimization(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=2,
        ordering_spec=ordering_spec,
        multi_spec=multi_spec,
        runtime_params=runtime_params,
    )

    assert len(results) == 2
    assert all(isinstance(r, RankedResult) for r in results)


def test_multi_start_ranks_soft_stage_only_results_by_final_score(monkeypatch) -> None:
    mol, selector, runtime_params = _build_runtime_case(dp=2)
    ordering_spec = _ordering_spec(max_candidates=4)
    multi_spec = MultiOptSpec(enabled=True, n_starts=3, top_k=3, seed=11, n_workers=1)
    scores = iter([1.5, 3.0, 2.0])

    def fake_run_single_start(*args, **kwargs):
        score = next(scores)
        return (
            score,
            mol.ToBinary(),
            {
                "final_score": score,
                "stage1_nonbonded_mode": "soft",
                "stage2_nonbonded_mode": None,
                "full_stage_skipped": True,
                "final_stage_nonbonded_mode": "soft",
            },
            0,
        )

    monkeypatch.setattr(
        "poly_csp.ordering.multi_opt._run_single_start",
        fake_run_single_start,
    )

    results = run_multi_start_optimization(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=2,
        ordering_spec=ordering_spec,
        multi_spec=multi_spec,
        runtime_params=runtime_params,
    )

    assert [result.score for result in results] == [3.0, 2.0, 1.5]
    assert all(result.summary["stage2_nonbonded_mode"] is None for result in results)
