from __future__ import annotations

import numpy as np

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorRegistry


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


def test_attach_selector_place_coords_false_builds_topology_without_conformer() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = build_backbone_structure(topology, _helix()).mol
    n_before = mol.GetNumAtoms()

    out = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
        place_coords=False,
    )

    assert out.GetNumConformers() == 0
    assert out.GetNumAtoms() > n_before


def test_attach_selector_without_input_conformer_keeps_topology_only() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = build_backbone_structure(topology, _helix()).mol
    mol.RemoveAllConformers()
    assert mol.GetNumConformers() == 0

    out = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
        place_coords=False,
    )

    assert out.GetNumConformers() == 0
    assert out.GetNumAtoms() > mol.GetNumAtoms()
