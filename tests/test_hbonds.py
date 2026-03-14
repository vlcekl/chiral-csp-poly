from __future__ import annotations

import numpy as np

from tests.support import build_backbone_coords
from tests.support import build_forcefield_mol
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.config.schema import HelixSpec
from poly_csp.ordering.hbonds import (
    compute_hbond_metrics,
    compute_selector_hbond_diagnostics,
    compute_selector_hbond_metrics,
)


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


def test_compute_hbond_metrics_runs_on_selector_decorated_polymer() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")
    dp = 3

    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol,
            residue_index=i,
            site="C6",
            selector=selector,
        )

    metrics = compute_hbond_metrics(mol=mol, selector=selector, max_distance_A=3.5)
    assert metrics.total_pairs >= 0
    assert metrics.donor_count == dp
    assert 0.0 <= metrics.like_fraction <= 1.0
    assert 0.0 <= metrics.geometric_fraction <= 1.0
    assert 0.0 <= metrics.like_donor_occupancy_fraction <= 1.0
    assert 0.0 <= metrics.geometric_donor_occupancy_fraction <= 1.0
    assert metrics.geometric_satisfied_pairs <= metrics.like_satisfied_pairs
    assert metrics.geometric_satisfied_donors <= metrics.like_satisfied_donors


def test_compute_selector_hbond_metrics_auto_uses_csp_connectivity_for_periodic_c6() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )

    metrics, applied_policy = compute_selector_hbond_metrics(
        mol=mol,
        selector=selector,
        connectivity_policy="auto",
    )

    assert applied_policy == "csp_literature_v1"
    assert metrics.donor_count == 4
    assert metrics.total_pairs == 4
    assert 0.0 <= metrics.geometric_fraction <= 1.0

    diagnostics = compute_selector_hbond_diagnostics(
        mol=mol,
        selector=selector,
        connectivity_policy="auto",
    )
    assert diagnostics.applied_policy == "csp_literature_v1"
    assert set(diagnostics.family_metrics) == {"c6_pitch_bridge"}
    assert diagnostics.family_metrics["c6_pitch_bridge"].total_pairs == 4


def test_compute_selector_hbond_metrics_auto_falls_back_when_no_target_edges_exist() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=3,
        selector=selector,
        site="C6",
        end_mode="open",
    )

    metrics, applied_policy = compute_selector_hbond_metrics(
        mol=mol,
        selector=selector,
        connectivity_policy="auto",
    )

    assert applied_policy == "generic"
    assert metrics.donor_count == 3

    diagnostics = compute_selector_hbond_diagnostics(
        mol=mol,
        selector=selector,
        connectivity_policy="auto",
    )
    assert diagnostics.applied_policy == "generic"
    assert diagnostics.family_metrics == {}
