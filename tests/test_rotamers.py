from __future__ import annotations

import pytest

from poly_csp.ordering.rotamers import default_rotamer_grid, enumerate_pose_library
from poly_csp.topology.selectors import SelectorRegistry


_CSP_SELECTOR_NAMES = ("35dcpc", "35dmpc", "3c4mpc", "3c5mpc", "4c3mpc", "5c2mpc")


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


@pytest.mark.parametrize("selector_name", _CSP_SELECTOR_NAMES)
def test_csp_selector_rotamer_catalog_is_consistent(selector_name: str) -> None:
    grid = default_rotamer_grid(SelectorRegistry.get(selector_name))

    assert set(grid.dihedral_values_deg) == {"tau_ar", "tau_link"}
    assert tuple(grid.dihedral_values_deg["tau_link"]) == (-120.0, -60.0, 60.0, 120.0)
    assert tuple(grid.dihedral_values_deg["tau_ar"]) == (-120.0, -60.0, 60.0, 120.0)
    assert grid.max_candidates == 64

    poses = enumerate_pose_library(grid)
    assert len(poses) == 16
    assert all(set(pose.dihedral_targets_deg) == {"tau_ar", "tau_link"} for pose in poses)


def test_tmb_rotamer_grid_follows_asset_definition() -> None:
    grid = default_rotamer_grid(SelectorRegistry.get("tmb"))

    assert set(grid.dihedral_values_deg) == {"tau_ar"}
    assert tuple(grid.dihedral_values_deg["tau_ar"]) == (-120.0, 0.0, 120.0)
    assert grid.max_candidates == 32

    poses = enumerate_pose_library(grid)
    assert len(poses) == 3
    assert all(set(pose.dihedral_targets_deg) == {"tau_ar"} for pose in poses)
