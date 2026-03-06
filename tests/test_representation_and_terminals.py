from __future__ import annotations

import json

import numpy as np

from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.terminals import apply_terminal_mode
from poly_csp.config.schema import HelixSpec


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


def test_natural_oh_polymerize_removes_internal_o1() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    dp = 4
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")

    n = template.mol.GetNumAtoms()
    assert mol.GetNumAtoms() == dp * n - (dp - 1)
    assert mol.GetProp("_poly_csp_polymer") == "amylose"
    assert mol.GetProp("_poly_csp_representation") == "natural_oh"

    maps = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
    assert len(maps) == dp
    assert "O1" in maps[0]
    for r in range(1, dp):
        assert "O1" not in maps[r]


def test_natural_oh_coords_pruning_and_selector_attachment() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()
    dp = 3

    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    removed = json.loads(mol.GetProp("_poly_csp_removed_old_indices_json"))
    keep_mask = np.ones((coords.shape[0],), dtype=bool)
    keep_mask[np.asarray(removed, dtype=int)] = False
    mol = assign_conformer(mol, coords[keep_mask])

    mol2 = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=1,
        site="C6",
        selector=selector,
    )
    assert mol2.GetNumConformers() == 1
    assert int(mol2.GetIntProp("_poly_csp_dp")) == dp


def test_apply_terminal_mode_sets_policy_metadata() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    n = mol.GetNumAtoms()

    open_mol = apply_terminal_mode(mol, mode="open", caps={}, representation="anhydro")
    capped_mol = apply_terminal_mode(
        mol, mode="capped", caps={"left": "acetyl", "right": "methoxy"}, representation="anhydro"
    )
    periodic_mol = apply_terminal_mode(
        mol, mode="periodic", caps={}, representation="anhydro"
    )

    assert open_mol.GetNumAtoms() == n
    assert capped_mol.GetNumAtoms() == n + 4
    assert periodic_mol.GetNumAtoms() == n

    assert open_mol.GetProp("_poly_csp_end_mode") == "open"
    assert capped_mol.GetProp("_poly_csp_end_mode") == "capped"
    assert periodic_mol.GetProp("_poly_csp_end_mode") == "periodic"

    assert open_mol.GetBoolProp("_poly_csp_terminal_topology_pending") is False
    assert capped_mol.GetBoolProp("_poly_csp_terminal_topology_pending") is False
    assert periodic_mol.GetBoolProp("_poly_csp_terminal_topology_pending") is False


def test_periodic_mode_closes_terminal_bond() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=3, linkage="1-4", anomer="alpha")
    periodic = apply_terminal_mode(mol, mode="periodic", caps={}, representation="anhydro")

    maps = json.loads(periodic.GetProp("_poly_csp_residue_label_map_json"))
    c1_0 = maps[0]["C1"]
    o4_last = maps[-1]["O4"]
    assert periodic.GetBondBetweenAtoms(c1_0, o4_last) is not None


def test_periodic_mode_natural_oh_removes_res0_o1() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    mol = polymerize(template=template, dp=3, linkage="1-4", anomer="alpha")
    n0 = mol.GetNumAtoms()

    periodic = apply_terminal_mode(
        mol,
        mode="periodic",
        caps={},
        representation="natural_oh",
    )

    maps = json.loads(periodic.GetProp("_poly_csp_residue_label_map_json"))
    assert "O1" not in maps[0]
    assert periodic.GetNumAtoms() == n0 - 1
    assert periodic.GetBondBetweenAtoms(maps[0]["C1"], maps[-1]["O4"]) is not None
