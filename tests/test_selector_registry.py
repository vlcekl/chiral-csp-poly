from __future__ import annotations

from poly_csp.topology.selector_assets import (
    available_selector_asset_names,
    load_selector_asset_spec,
)
from poly_csp.topology.selectors import SelectorRegistry, selector_from_smiles


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


def test_selector_from_smiles_detects_implicit_h_donors() -> None:
    tpl = selector_from_smiles(
        name="implicit_nh",
        smiles="[*:1][C:2](=[O:3])[NH:4][c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1",
        attach_atom_idx=1,
        attach_dummy_idx=0,
        dihedrals={},
    )
    assert tpl.donors
