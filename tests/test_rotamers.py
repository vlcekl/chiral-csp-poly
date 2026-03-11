from __future__ import annotations

from poly_csp.ordering.rotamers import default_rotamer_grid, enumerate_pose_library
from poly_csp.topology.selectors import SelectorRegistry


def test_default_rotamer_grid_has_expected_keys_for_dmpc() -> None:
    grid = default_rotamer_grid(SelectorRegistry.get("35dmpc"))
    assert "tau_link" in grid.dihedral_values_deg
    assert "tau_ar" in grid.dihedral_values_deg
    assert grid.max_candidates > 0


def test_enumerate_pose_library_is_deterministic_and_nonempty() -> None:
    grid = default_rotamer_grid(SelectorRegistry.get("35dmpc"))
    poses1 = enumerate_pose_library(grid)
    poses2 = enumerate_pose_library(grid)
    assert len(poses1) > 0
    assert len(poses1) == len(poses2)
    assert poses1[0].dihedral_targets_deg == poses2[0].dihedral_targets_deg
