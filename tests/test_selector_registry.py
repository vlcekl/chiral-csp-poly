from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from poly_csp.topology.selector_assets import (
    available_selector_asset_names,
    load_selector_asset_spec,
)
from poly_csp.topology.selectors import SelectorRegistry, selector_from_smiles


_ROOT = Path(__file__).resolve().parents[1]


def test_selector_registry_register_and_get() -> None:
    got = SelectorRegistry.get("35dmpc")
    assert got.name == "35dmpc"
    assert got.attach_atom_idx is not None


def test_selector_asset_catalog_lists_current_assets() -> None:
    expected = {
        "35dcpc",
        "35dmpc",
        "3c4mpc",
        "3c5mpc",
        "4c3mpc",
        "5c2mpc",
        "tmb",
    }
    assert set(available_selector_asset_names()) == expected
    assert set(SelectorRegistry.available()) == expected
    spec = load_selector_asset_spec("35dmpc")
    assert spec.name == "35dmpc"
    assert spec.linkage_type == "carbamate"
    assert spec.reference_columns == ("AD", "IB")
    assert spec.reference_backbones == ("amylose", "cellulose")


@pytest.mark.parametrize("selector_name", available_selector_asset_names())
def test_selector_registry_templates_expose_runtime_attachment_contract(
    selector_name: str,
) -> None:
    template = SelectorRegistry.get(selector_name)

    assert 0 <= template.attach_atom_idx < template.mol.GetNumAtoms()
    assert template.attach_dummy_idx is not None
    assert 0 <= template.attach_dummy_idx < template.mol.GetNumAtoms()
    assert template.attach_atom_idx != template.attach_dummy_idx
    assert template.connector_local_roles
    assert all(
        0 <= atom_idx < template.mol.GetNumAtoms()
        for atom_idx in template.connector_local_roles
    )
    assert all(len(indices) == 4 for indices in template.dihedrals.values())
    assert all(
        0 <= atom_idx < template.mol.GetNumAtoms()
        for indices in template.dihedrals.values()
        for atom_idx in indices
    )
    if template.linkage_type == "carbamate":
        assert set(template.connector_local_roles.values()) == {
            "carbonyl_c",
            "carbonyl_o",
            "amide_n",
        }
    elif template.linkage_type == "ester":
        assert set(template.connector_local_roles.values()) == {
            "carbonyl_c",
            "carbonyl_o",
        }


def test_selector_registry_rejects_removed_legacy_alias() -> None:
    assert "dmpc_35" not in SelectorRegistry.available()
    with pytest.raises(KeyError, match="Unknown selector"):
        SelectorRegistry.get("dmpc_35")


def test_selector_asset_loading_is_silent() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from poly_csp.topology.selector_assets import "
                "available_selector_asset_names, load_selector_asset_template; "
                "[load_selector_asset_template(name) for name in available_selector_asset_names()]"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env=env,
    )

    assert "UFFTYPER" not in result.stderr


def test_selector_from_smiles_detects_implicit_h_donors() -> None:
    tpl = selector_from_smiles(
        name="implicit_nh",
        smiles="[*:1][C:2](=[O:3])[NH:4][c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1",
        attach_atom_idx=1,
        attach_dummy_idx=0,
        dihedrals={},
    )
    assert tpl.donors
