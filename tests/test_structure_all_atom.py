from __future__ import annotations

import json

import numpy as np
import pytest

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.backbone_builder import (
    _linkage_targets,
    build_backbone_structure,
    inspect_backbone_linkages,
)
from poly_csp.structure.pbc import compute_helical_box_vectors, set_box_vectors
from tests.support import build_backbone_coords
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.residue_state import resolve_residue_template_states
from poly_csp.structure.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.terminals import apply_terminal_mode


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def _expected_heavy_coords(template, topology_mol, dp: int) -> np.ndarray:
    coords = build_backbone_coords(template, _helix(), dp)
    removed = json.loads(topology_mol.GetProp("_poly_csp_removed_old_indices_json"))
    if removed:
        keep_mask = np.ones((coords.shape[0],), dtype=bool)
        keep_mask[np.asarray(removed, dtype=int)] = False
        coords = coords[keep_mask]
    return coords


def _periodic_helix() -> HelixSpec:
    return HelixSpec(
        name="periodic_test_helix",
        theta_rad=-4.71238898038469,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=4.933333333333334,
        handedness="left",
    )


def test_select_residue_templates_detects_substitution_and_termini() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()

    topology = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = build_backbone_structure(topology, _helix()).mol
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    states = resolve_residue_template_states(mol)
    assert len(states) == 2
    assert states[0].substituted_sites == ("C6",)
    assert states[0].incoming_link is False
    assert states[0].outgoing_link is True
    assert states[1].incoming_link is True
    assert states[1].outgoing_link is False


def test_select_residue_templates_tracks_periodic_and_capped_anchor_states() -> None:
    natural = make_glucose_template("amylose", monomer_representation="natural_oh")
    periodic = polymerize(template=natural, dp=3, linkage="1-4", anomer="alpha")
    periodic = apply_terminal_mode(
        periodic,
        mode="periodic",
        caps={},
        representation="natural_oh",
    )
    periodic_states = resolve_residue_template_states(periodic)
    assert periodic_states[0].has_o1 is False
    assert all(state.incoming_link for state in periodic_states)
    assert all(state.outgoing_link for state in periodic_states)

    anhydro = make_glucose_template("amylose")
    capped = polymerize(template=anhydro, dp=2, linkage="1-4", anomer="alpha")
    capped = apply_terminal_mode(
        capped,
        mode="capped",
        caps={"left": "acetyl", "right": "methoxy"},
        representation="anhydro",
    )
    capped_states = resolve_residue_template_states(capped)
    assert capped_states[0].left_cap == "acetyl"
    assert capped_states[0].left_anchor_label == "C1"
    assert capped_states[-1].right_cap == "methoxy"
    assert capped_states[-1].right_anchor_label == "O4"


def test_build_backbone_structure_places_explicit_h_backbone_directly() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    topology = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")

    out = build_backbone_structure(topology, _helix()).mol
    maps = json.loads(out.GetProp("_poly_csp_residue_label_map_json"))
    all_xyz = np.asarray(out.GetConformer(0).GetPositions(), dtype=float)
    expected_heavy = _expected_heavy_coords(template, topology, dp=2)
    heavy_indices = sorted(
        atom_idx
        for mapping in maps
        for atom_idx in mapping.values()
    )

    assert np.allclose(all_xyz[heavy_indices], expected_heavy, atol=1e-6)

    res0_o4 = out.GetAtomWithIdx(maps[0]["O4"])
    res1_o4 = out.GetAtomWithIdx(maps[1]["O4"])
    res0_c6 = out.GetAtomWithIdx(maps[0]["C6"])

    assert sum(1 for nbr in res0_o4.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert sum(1 for nbr in res1_o4.GetNeighbors() if nbr.GetAtomicNum() == 1) == 1
    assert sum(1 for nbr in res0_c6.GetNeighbors() if nbr.GetAtomicNum() == 1) == 2


def test_build_backbone_structure_preserves_glycosidic_linkage_geometry() -> None:
    topology = polymerize(
        template=make_glucose_template("amylose"),
        dp=4,
        linkage="1-4",
        anomer="alpha",
    )
    out = build_backbone_structure(topology, _helix()).mol
    targets = _linkage_targets(polymer="amylose", representation="anhydro")
    metrics = inspect_backbone_linkages(out)

    assert len(metrics) == 3
    for metric in metrics:
        assert abs(metric.bond_length_A - targets.bond_length_A) < 0.12
        assert abs(metric.donor_angle_deg - targets.donor_angle_deg) < 10.0
        assert abs(metric.acceptor_angle_deg - targets.acceptor_angle_deg) < 10.0
        assert abs(metric.acceptor_angle_c2_deg - targets.acceptor_angle_c2_deg) < 10.0


def test_build_backbone_structure_keeps_incoming_o4_out_of_h1_slot() -> None:
    topology = polymerize(
        template=make_glucose_template("amylose"),
        dp=3,
        linkage="1-4",
        anomer="alpha",
    )
    out = build_backbone_structure(topology, _helix()).mol
    metrics = inspect_backbone_linkages(out)

    assert len(metrics) == 2
    assert all(metric.o4_h1_distance_A is not None for metric in metrics)
    assert all(metric.o4_h1_distance_A > 1.4 for metric in metrics if metric.o4_h1_distance_A is not None)


def test_build_backbone_structure_rejects_incompatible_helix_request() -> None:
    topology = polymerize(
        template=make_glucose_template("amylose"),
        dp=3,
        linkage="1-4",
        anomer="alpha",
    )
    bad_helix = HelixSpec(
        name="chemically_bad",
        theta_rad=0.5,
        rise_A=12.0,
        repeat_residues=1,
        repeat_turns=1,
        residues_per_turn=1.0,
        pitch_A=12.0,
        handedness="left",
    )

    with pytest.raises(
        ValueError,
        match="chemically plausible glycosidic bond",
    ):
        build_backbone_structure(topology, bad_helix)


def test_selector_attachment_uses_explicit_h_backbone_and_selector_templates() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()

    topology = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = build_backbone_structure(topology, _helix()).mol
    maps = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
    expected_heavy = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)

    out = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    all_xyz = np.asarray(out.GetConformer(0).GetPositions(), dtype=float)
    heavy_indices = sorted(
        atom_idx
        for mapping in maps
        for atom_idx in mapping.values()
    )
    assert np.allclose(all_xyz[heavy_indices], expected_heavy[heavy_indices], atol=1e-6)

    res0_o6 = out.GetAtomWithIdx(maps[0]["O6"])
    assert sum(1 for nbr in res0_o6.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0

    amide_n = next(
        atom
        for atom in out.GetAtoms()
        if atom.HasProp("_poly_csp_connector_role")
        and atom.GetProp("_poly_csp_connector_role") == "amide_n"
    )
    amide_h = [nbr for nbr in amide_n.GetNeighbors() if nbr.GetAtomicNum() == 1]
    assert len(amide_h) == 1
    assert amide_h[0].GetProp("_poly_csp_component") == "connector"


def test_build_backbone_structure_handles_capped_termini() -> None:
    template = make_glucose_template("amylose")
    topology = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    topology = apply_terminal_mode(
        topology,
        mode="capped",
        caps={"left": "acetyl", "right": "methoxy"},
        representation="anhydro",
    )

    out = build_backbone_structure(topology, _helix()).mol
    maps = json.loads(out.GetProp("_poly_csp_residue_label_map_json"))
    c1 = out.GetAtomWithIdx(maps[0]["C1"])
    o4 = out.GetAtomWithIdx(maps[-1]["O4"])

    assert sum(1 for nbr in c1.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert sum(1 for nbr in o4.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert json.loads(out.GetProp("_poly_csp_terminal_cap_indices_json"))["left"]
    assert json.loads(out.GetProp("_poly_csp_terminal_cap_indices_json"))["right"]


def test_build_backbone_structure_assigns_coordinates_without_input_conformer() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    topology = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    topology.RemoveAllConformers()

    out = build_backbone_structure(topology, _helix()).mol
    assert out.GetNumConformers() == 1
    assert out.GetNumAtoms() > topology.GetNumAtoms()


def test_build_backbone_structure_rejects_non_commensurate_periodic_helix() -> None:
    template = make_glucose_template("amylose", monomer_representation="anhydro")
    topology = polymerize(template=template, dp=3, linkage="1-4", anomer="alpha")
    topology = apply_terminal_mode(
        topology,
        mode="periodic",
        caps={},
        representation="anhydro",
    )

    with pytest.raises(ValueError, match="closes the screw rotation"):
        build_backbone_structure(topology, _periodic_helix())


def test_inspect_backbone_linkages_uses_minimum_image_for_periodic_closure() -> None:
    template = make_glucose_template("amylose", monomer_representation="anhydro")
    topology = polymerize(template=template, dp=4, linkage="1-4", anomer="alpha")
    topology = apply_terminal_mode(
        topology,
        mode="periodic",
        caps={},
        representation="anhydro",
    )
    out = build_backbone_structure(topology, _periodic_helix()).mol
    Lx_A, Ly_A, Lz_A = compute_helical_box_vectors(out, _periodic_helix(), dp=4, padding_A=30.0)
    set_box_vectors(out, Lx_A, Ly_A, Lz_A)

    metrics = inspect_backbone_linkages(out)

    assert len(metrics) == 4
    closure = metrics[-1]
    assert closure.donor_residue_index == 3
    assert closure.acceptor_residue_index == 0
    assert closure.bond_length_A < 2.0
