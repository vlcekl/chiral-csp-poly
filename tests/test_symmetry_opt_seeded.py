"""Tests for seeded optimizer diversity and determinism."""
from __future__ import annotations

import numpy as np

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.functionalization import attach_selector
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.chemistry.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.config.schema import HelixSpec
from poly_csp.ordering.symmetry_opt import OrderingSpec, optimize_selector_ordering


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


def test_seeded_determinism() -> None:
    mol, selector = _build_mol(dp=3)
    spec = OrderingSpec(enabled=True, repeat_residues=1, max_candidates=8)

    _, summary1 = optimize_selector_ordering(
        mol=mol, selector=selector, sites=["C6"], dp=3, spec=spec, seed=42,
    )
    _, summary2 = optimize_selector_ordering(
        mol=mol, selector=selector, sites=["C6"], dp=3, spec=spec, seed=42,
    )

    assert summary1["final_score"] == summary2["final_score"]
    assert summary1["seed"] == summary2["seed"] == 42


def test_different_seeds_produce_seed_metadata() -> None:
    mol, selector = _build_mol(dp=3)
    spec = OrderingSpec(enabled=True, repeat_residues=1, max_candidates=8)

    _, summary_a = optimize_selector_ordering(
        mol=mol, selector=selector, sites=["C6"], dp=3, spec=spec, seed=42,
    )
    _, summary_b = optimize_selector_ordering(
        mol=mol, selector=selector, sites=["C6"], dp=3, spec=spec, seed=99,
    )

    assert summary_a["seed"] == 42
    assert summary_b["seed"] == 99


def test_no_seed_backward_compatible() -> None:
    mol, selector = _build_mol(dp=3)
    spec = OrderingSpec(enabled=True, repeat_residues=1, max_candidates=8)

    _, summary = optimize_selector_ordering(
        mol=mol, selector=selector, sites=["C6"], dp=3, spec=spec,
    )

    assert summary["seed"] is None
    assert "final_score" in summary
