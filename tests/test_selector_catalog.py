from __future__ import annotations

import csv
from pathlib import Path

import pytest

from poly_csp.topology.selector_assets import (
    available_selector_asset_names,
    load_selector_asset_spec,
)
from poly_csp.topology.selectors import SelectorRegistry


_CSP_SELECTOR_EXPECTATIONS = {
    "35dmpc": {
        "full_name": "3,5-dimethylphenylcarbamate",
        "reference_columns": ("AD", "IB"),
        "reference_backbones": ("amylose", "cellulose"),
    },
    "5c2mpc": {
        "full_name": "5-chloro-2-methylphenylcarbamate",
        "reference_columns": ("AY",),
        "reference_backbones": ("amylose",),
    },
    "35dcpc": {
        "full_name": "3,5-dichlorophenylcarbamate",
        "reference_columns": ("IC", "IE"),
        "reference_backbones": ("amylose", "cellulose"),
    },
    "3c4mpc": {
        "full_name": "3-chloro-4-methylphenylcarbamate",
        "reference_columns": ("IF", "OZ"),
        "reference_backbones": ("amylose", "cellulose"),
    },
    "3c5mpc": {
        "full_name": "3-chloro-5-methylphenylcarbamate",
        "reference_columns": ("IG",),
        "reference_backbones": ("amylose",),
    },
    "4c3mpc": {
        "full_name": "4-chloro-3-methylphenylcarbamate",
        "reference_columns": ("OX",),
        "reference_backbones": ("cellulose",),
    },
}


@pytest.mark.parametrize("selector_name", sorted(_CSP_SELECTOR_EXPECTATIONS))
def test_csp_selector_assets_load_with_expected_metadata(selector_name: str) -> None:
    expected = _CSP_SELECTOR_EXPECTATIONS[selector_name]
    spec = load_selector_asset_spec(selector_name)
    template = SelectorRegistry.get(selector_name)

    assert spec.name == selector_name
    assert spec.full_name == expected["full_name"]
    assert spec.reference_columns == expected["reference_columns"]
    assert spec.reference_backbones == expected["reference_backbones"]
    assert spec.linkage_type == "carbamate"
    assert set(spec.dihedrals) == {"tau_link", "tau_ar", "tau_ring"}
    assert set(spec.connector_role_by_map_num.values()) == {
        "carbonyl_c",
        "carbonyl_o",
        "amide_n",
    }
    assert template.full_name == expected["full_name"]
    assert template.reference_columns == expected["reference_columns"]
    assert template.reference_backbones == expected["reference_backbones"]
    assert template.linkage_type == "carbamate"
    assert set(template.dihedrals) == {"tau_link", "tau_ar", "tau_ring"}
    assert set(template.connector_local_roles.values()) == {
        "carbonyl_c",
        "carbonyl_o",
        "amide_n",
    }
    assert set(template.rotamer_grid) == {"tau_link", "tau_ar"}
    assert template.rotamer_max_candidates == 64
    assert template.donors
    assert template.acceptors


def test_csp_selector_catalog_matches_reference_csv_unique_selectors() -> None:
    reference_csv = Path("docs/columns/columns_extended.csv")
    with reference_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    expected_selector_names = {
        row["Selector Molecule"].strip()
        for row in rows
        if row["Column ID"].strip().upper() != "WO"
    }
    catalog_selector_names = {
        load_selector_asset_spec(name).full_name
        for name in available_selector_asset_names()
        if name != "tmb"
    }

    assert catalog_selector_names == expected_selector_names


def test_csp_selector_reference_columns_cover_all_polymer_phase_rows() -> None:
    reference_csv = Path("docs/columns/columns_extended.csv")
    with reference_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    expected_columns = {
        row["Column ID"].strip()
        for row in rows
        if row["Column ID"].strip().upper() != "WO"
    }
    catalog_columns = {
        column_id
        for selector_name in _CSP_SELECTOR_EXPECTATIONS
        for column_id in load_selector_asset_spec(selector_name).reference_columns
    }

    assert catalog_columns == expected_columns
