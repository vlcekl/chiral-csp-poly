from __future__ import annotations

import numpy as np

from tests.support import build_backbone_coords
from poly_csp.topology.reactions import (
    attach_selector,
)
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.config.schema import HelixSpec, SelectorPoseSpec
from poly_csp.structure.alignment import apply_selector_pose_dihedrals
from poly_csp.structure.dihedrals import measure_dihedral_rad, set_dihedral_rad


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


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def test_set_dihedral_rad_changes_only_masked_atoms() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 1.0],  # a
            [0.0, 0.0, 0.0],  # b
            [1.0, 0.0, 0.0],  # c
            [1.0, 1.0, 0.0],  # d
            [2.0, 1.0, 1.0],  # downstream
        ],
        dtype=float,
    )
    current = measure_dihedral_rad(coords, 0, 1, 2, 3)
    target = current + np.deg2rad(60.0)

    mask = np.array([False, False, True, True, True], dtype=bool)
    out = set_dihedral_rad(coords, 0, 1, 2, 3, target, mask)

    assert np.allclose(out[0], coords[0], atol=1e-12)
    assert np.allclose(out[1], coords[1], atol=1e-12)
    assert not np.allclose(out[4], coords[4], atol=1e-6)

    measured = measure_dihedral_rad(out, 0, 1, 2, 3)
    assert abs(_angle_diff(measured, target)) < 1e-8


def test_apply_selector_pose_dihedrals_sets_target_angle() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    coords = build_backbone_coords(template, _helix(), dp=1)
    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    pose = SelectorPoseSpec(dihedral_targets_deg={"tau_ar": 45.0})
    out = apply_selector_pose_dihedrals(
        mol=mol,
        residue_index=0,
        site="C6",
        pose_spec=pose,
        selector=selector,
    )

    local_to_global = {}
    for atom in out.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if int(atom.GetIntProp("_poly_csp_residue_index")) != 0:
            continue
        if atom.GetProp("_poly_csp_site") != "C6":
            continue
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        local_to_global[local] = atom.GetIdx()

    a_l, b_l, c_l, d_l = selector.dihedrals["tau_ar"]
    a, b, c, d = (
        local_to_global[a_l],
        local_to_global[b_l],
        local_to_global[c_l],
        local_to_global[d_l],
    )
    xyz = np.asarray(out.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    measured = measure_dihedral_rad(xyz, a, b, c, d)
    target = np.deg2rad(45.0)
    assert abs(_angle_diff(measured, target)) < 1e-6
