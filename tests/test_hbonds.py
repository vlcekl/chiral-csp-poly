from __future__ import annotations

import numpy as np
from rdkit import Chem

from poly_csp.forcefield.model import build_forcefield_molecule
from tests.support import build_backbone_coords
from tests.support import build_forcefield_mol
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.config.schema import HelixSpec
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.terminals import apply_terminal_mode
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


def _build_multisite_forcefield_mol(
    *,
    polymer: str,
    dp: int,
    selector_name: str,
    sites: list[str],
    end_mode: str = "open",
) -> Chem.Mol:
    selector = SelectorRegistry.get(selector_name)
    template = make_glucose_template(polymer, monomer_representation="anhydro")
    topology = polymerize(
        template=template,
        dp=dp,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    topology = apply_terminal_mode(
        mol=topology,
        mode=end_mode,  # type: ignore[arg-type]
        caps={},
        representation="anhydro",
    )
    structure = build_backbone_structure(topology, _helix()).mol
    for residue_index in range(dp):
        for site in sites:
            structure = attach_selector(
                mol_polymer=structure,
                residue_index=residue_index,
                site=site,
                selector=selector,
            )
    return build_forcefield_molecule(structure).mol


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


def test_compute_selector_hbond_metrics_auto_uses_custom_connectivity_for_periodic_c6() -> None:
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

    assert applied_policy == "custom_v1"
    assert metrics.donor_count == 4
    assert metrics.total_pairs == 4
    assert 0.0 <= metrics.geometric_fraction <= 1.0

    diagnostics = compute_selector_hbond_diagnostics(
        mol=mol,
        selector=selector,
        connectivity_policy="auto",
    )
    assert diagnostics.applied_policy == "custom_v1"
    assert set(diagnostics.family_metrics) == {"c6_forward_neighbor"}
    assert diagnostics.family_metrics["c6_forward_neighbor"].total_pairs == 4


def test_compute_selector_hbond_metrics_auto_uses_custom_connectivity_for_open_c6() -> None:
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

    assert applied_policy == "custom_v1"
    assert metrics.donor_count == 2
    assert metrics.total_pairs == 2

    diagnostics = compute_selector_hbond_diagnostics(
        mol=mol,
        selector=selector,
        connectivity_policy="auto",
    )
    assert diagnostics.applied_policy == "custom_v1"
    assert set(diagnostics.family_metrics) == {"c6_forward_neighbor"}
    assert diagnostics.family_metrics["c6_forward_neighbor"].total_pairs == 2


def test_compute_selector_hbond_metrics_custom_v1_uses_forward_c3_to_c2_edges() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = _build_multisite_forcefield_mol(
        polymer="amylose",
        dp=3,
        selector_name="35dmpc",
        sites=["C2", "C3"],
        end_mode="open",
    )

    diagnostics = compute_selector_hbond_diagnostics(
        mol=mol,
        selector=selector,
        connectivity_policy="custom_v1",
    )

    assert diagnostics.applied_policy == "custom_v1"
    assert set(diagnostics.family_metrics) == {"c3_to_c2_forward_neighbor"}
    assert diagnostics.family_metrics["c3_to_c2_forward_neighbor"].total_pairs == 2
