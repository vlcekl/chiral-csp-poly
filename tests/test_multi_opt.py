"""Tests for multi-start selector optimization."""
from __future__ import annotations

import numpy as np

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.functionalization import attach_selector
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.chemistry.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.config.schema import HelixSpec
from poly_csp.ordering.multi_opt import MultiOptSpec, RankedResult, run_multi_start_optimization
from poly_csp.ordering.symmetry_opt import OrderingSpec


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0 * np.pi / 2.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def _build_mol(dp: int = 3):
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol,
            template=template,
            residue_index=i,
            site="C6",
            selector=selector,
        )
    return mol, selector


def test_multi_start_returns_ranked_results() -> None:
    mol, selector = _build_mol(dp=3)
    ordering_spec = OrderingSpec(enabled=True, repeat_residues=1, max_candidates=8)
    multi_spec = MultiOptSpec(enabled=True, n_starts=3, top_k=2, seed=42)

    results = run_multi_start_optimization(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=3,
        ordering_spec=ordering_spec,
        multi_spec=multi_spec,
    )

    assert len(results) == 2
    assert all(isinstance(r, RankedResult) for r in results)
    assert results[0].rank == 1
    assert results[1].rank == 2
    # Scores should be descending.
    assert results[0].score >= results[1].score
    # Mol should have correct atom count.
    assert results[0].mol.GetNumAtoms() == mol.GetNumAtoms()


def test_multi_start_reproducibility() -> None:
    mol, selector = _build_mol(dp=3)
    ordering_spec = OrderingSpec(enabled=True, repeat_residues=1, max_candidates=8)
    multi_spec = MultiOptSpec(enabled=True, n_starts=3, top_k=2, seed=123)

    run1 = run_multi_start_optimization(
        mol=mol, selector=selector, sites=["C6"], dp=3,
        ordering_spec=ordering_spec, multi_spec=multi_spec,
    )
    run2 = run_multi_start_optimization(
        mol=mol, selector=selector, sites=["C6"], dp=3,
        ordering_spec=ordering_spec, multi_spec=multi_spec,
    )

    assert len(run1) == len(run2)
    for r1, r2 in zip(run1, run2):
        assert r1.score == r2.score
        assert r1.seed_used == r2.seed_used
        assert r1.rank == r2.rank


def test_multi_start_no_seed_runs() -> None:
    mol, selector = _build_mol(dp=3)
    ordering_spec = OrderingSpec(enabled=True, repeat_residues=1, max_candidates=4)
    multi_spec = MultiOptSpec(enabled=True, n_starts=2, top_k=2, seed=None)

    results = run_multi_start_optimization(
        mol=mol, selector=selector, sites=["C6"], dp=3,
        ordering_spec=ordering_spec, multi_spec=multi_spec,
    )

    assert len(results) == 2
    assert results[0].rank == 1
