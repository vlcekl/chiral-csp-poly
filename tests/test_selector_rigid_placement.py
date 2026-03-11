from __future__ import annotations

import numpy as np

from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.config.schema import SelectorPoseSpec
from poly_csp.structure.local_frames import compute_residue_local_frame, pose_selector_in_frame


def test_selector_pose_is_deterministic() -> None:
    mono = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    coords_res = np.asarray(mono.mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    coords_sel = np.asarray(selector.mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))

    r, t = compute_residue_local_frame(coords_res, mono.atom_idx)
    pose = SelectorPoseSpec(carbonyl_dir_local=(-1.0, 0.2, 0.1))

    p1 = pose_selector_in_frame(coords_sel, pose=pose, r_res=r, t_res=t, attach_atom_idx=selector.attach_atom_idx)
    p2 = pose_selector_in_frame(coords_sel, pose=pose, r_res=r, t_res=t, attach_atom_idx=selector.attach_atom_idx)
    assert np.allclose(p1, p2, atol=1e-12)


def test_selector_pose_has_no_catastrophic_overlap_single_residue() -> None:
    mono = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    coords_res = np.asarray(mono.mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    coords_sel = np.asarray(selector.mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))

    r, t = compute_residue_local_frame(coords_res, mono.atom_idx)
    pose = SelectorPoseSpec(carbonyl_dir_local=(-1.0, 0.0, 0.0))
    placed = pose_selector_in_frame(coords_sel, pose=pose, r_res=r, t_res=t, attach_atom_idx=selector.attach_atom_idx)

    # Exclude the exact attachment coincidence by skipping the closest pair.
    diffs = coords_res[:, None, :] - placed[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2).ravel()
    d = np.sqrt(np.sort(d2))
    assert d[1] > 0.5
