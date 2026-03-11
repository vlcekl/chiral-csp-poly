from __future__ import annotations

import csv
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


_ROOT = Path(__file__).resolve().parents[1]
_CONF_DIR = _ROOT / "conf"
_PHASE_PRESET_BY_COLUMN = {
    "AD": "chiralpak_ad",
    "AY": "chiralpak_ay",
    "IB": "chiralpak_ib",
    "IC": "chiralpak_ic",
    "IE": "chiralpak_ie",
    "IF": "chiralpak_if",
    "IG": "chiralpak_ig",
    "OX": "chiralcel_ox",
    "OZ": "chiralcel_oz",
}
_EXPECTED_PHASE_COMPOSITION = {
    "chiralpak_ad": {
        "column_id": "AD",
        "polymer": "amylose",
        "selector": "35dmpc",
        "helix_name": "amylose_CSP_4_3_derivatized",
        "attachment_mode": "coated",
    },
    "chiralpak_ay": {
        "column_id": "AY",
        "polymer": "amylose",
        "selector": "5c2mpc",
        "helix_name": "amylose_CSP_4_3_derivatized",
        "attachment_mode": "coated",
    },
    "chiralpak_ib": {
        "column_id": "IB",
        "polymer": "cellulose",
        "selector": "35dmpc",
        "helix_name": "cellulose_CSP_3_2_derivatized",
        "attachment_mode": "immobilized",
    },
    "chiralpak_ic": {
        "column_id": "IC",
        "polymer": "cellulose",
        "selector": "35dcpc",
        "helix_name": "cellulose_CSP_3_2_derivatized",
        "attachment_mode": "immobilized",
    },
    "chiralpak_ie": {
        "column_id": "IE",
        "polymer": "amylose",
        "selector": "35dcpc",
        "helix_name": "amylose_CSP_4_3_derivatized",
        "attachment_mode": "immobilized",
    },
    "chiralpak_if": {
        "column_id": "IF",
        "polymer": "amylose",
        "selector": "3c4mpc",
        "helix_name": "amylose_CSP_4_3_derivatized",
        "attachment_mode": "immobilized",
    },
    "chiralpak_ig": {
        "column_id": "IG",
        "polymer": "amylose",
        "selector": "3c5mpc",
        "helix_name": "amylose_CSP_4_3_derivatized",
        "attachment_mode": "immobilized",
    },
    "chiralcel_ox": {
        "column_id": "OX",
        "polymer": "cellulose",
        "selector": "4c3mpc",
        "helix_name": "cellulose_CSP_3_2_derivatized",
        "attachment_mode": "coated",
    },
    "chiralcel_oz": {
        "column_id": "OZ",
        "polymer": "cellulose",
        "selector": "3c4mpc",
        "helix_name": "cellulose_CSP_3_2_derivatized",
        "attachment_mode": "coated",
    },
}


def _compose_phase_config(phase_preset: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="config", overrides=[f"phase={phase_preset}"])


def test_default_config_stays_generic_and_not_phase_bound() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name="config")

    assert "phase" not in cfg or cfg.phase is None
    assert cfg.topology.backbone.kind == "amylose"
    assert cfg.topology.selector.name == "35dmpc"
    assert cfg.structure.helix.name == "amylose_CSP_4_3_derivatized"


@pytest.mark.parametrize("phase_preset", sorted(_EXPECTED_PHASE_COMPOSITION))
def test_phase_preset_composes_expected_groups(phase_preset: str) -> None:
    expected = _EXPECTED_PHASE_COMPOSITION[phase_preset]
    cfg = _compose_phase_config(phase_preset)

    assert cfg.phase.column_id == expected["column_id"]
    assert cfg.phase.attachment_mode == expected["attachment_mode"]
    assert cfg.topology.backbone.kind == expected["polymer"]
    assert cfg.topology.selector.name == expected["selector"]
    assert cfg.structure.helix.name == expected["helix_name"]


def test_phase_presets_cover_all_non_wo_reference_rows() -> None:
    reference_csv = _ROOT / "docs/columns/columns_extended.csv"
    with reference_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    expected_columns = {
        row["Column ID"].strip()
        for row in rows
        if row["Column ID"].strip().upper() != "WO"
    }
    assert set(_PHASE_PRESET_BY_COLUMN) == expected_columns

    for row in rows:
        column_id = row["Column ID"].strip()
        if column_id.upper() == "WO":
            continue
        cfg = _compose_phase_config(_PHASE_PRESET_BY_COLUMN[column_id])
        assert cfg.phase.column_id == column_id
        assert cfg.phase.manufacturer == row["Manufacturer"].strip()
        assert cfg.phase.chemical_name == row["Chemical Name"].strip()
        assert cfg.phase.attachment_description == row["Mode of Attachment"].strip()
        assert cfg.phase.silica_tether_description == row["Tether/Linkage to Silica"].strip()
