from __future__ import annotations

import numpy as np

from poly_csp.config.schema import SelectorPoseSpec
from poly_csp.structure.alignment import (
    _resolve_selector_dihedral_atom_indices,
    apply_selector_pose_dihedrals,
)
from poly_csp.structure.dihedrals import measure_dihedral_rad
from poly_csp.topology.selectors import SelectorRegistry
from tests.support import build_forcefield_mol


def test_apply_selector_pose_dihedrals_supports_anchor_dihedrals() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    a, b, c, d = _resolve_selector_dihedral_atom_indices(
        mol,
        0,
        "C6",
        selector,
        "tau_attach",
    )
    xyz_before = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    before_deg = float(np.rad2deg(measure_dihedral_rad(xyz_before, a, b, c, d)))

    updated = apply_selector_pose_dihedrals(
        mol,
        0,
        "C6",
        SelectorPoseSpec(dihedral_targets_deg={"tau_attach": 120.0}),
        selector,
    )

    xyz_after = np.asarray(updated.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    after_deg = float(np.rad2deg(measure_dihedral_rad(xyz_after, a, b, c, d)))
    assert abs(after_deg - 120.0) < 1.0e-4
    assert abs(after_deg - before_deg) > 1.0
