from __future__ import annotations

import math

import pytest

from poly_csp.config.schema import HelixSpec


def test_helix_spec_derives_fields_from_repeat_and_axial_repeat() -> None:
    helix = HelixSpec(
        name="amylose_4_3_reference",
        repeat_residues=4,
        repeat_turns=3,
        axial_repeat_A=14.6,
        handedness="left",
    )

    assert helix.theta_rad == pytest.approx(-3.0 * math.pi / 2.0)
    assert helix.rise_A == pytest.approx(3.65)
    assert helix.residues_per_turn == pytest.approx(4.0 / 3.0)
    assert helix.pitch_A == pytest.approx(14.6 / 3.0)
    assert helix.axial_repeat_A == pytest.approx(14.6)


def test_helix_spec_derives_cellulose_csp_3_2_geometry() -> None:
    helix = HelixSpec(
        name="cellulose_3_2_reference",
        repeat_residues=3,
        repeat_turns=2,
        axial_repeat_A=16.2,
        handedness="left",
    )

    assert helix.theta_rad == pytest.approx(-4.0 * math.pi / 3.0)
    assert helix.rise_A == pytest.approx(5.4)
    assert helix.residues_per_turn == pytest.approx(1.5)
    assert helix.pitch_A == pytest.approx(8.1)
    assert helix.axial_repeat_A == pytest.approx(16.2)


def test_helix_spec_derives_axial_repeat_from_legacy_fields() -> None:
    helix = HelixSpec(
        name="legacy_low_level_amylose",
        theta_rad=-3.0 * math.pi / 2.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )

    assert helix.axial_repeat_A == pytest.approx(14.8)


def test_helix_spec_rejects_inconsistent_repeat_derived_fields() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        HelixSpec(
            name="bad_amylose",
            repeat_residues=4,
            repeat_turns=3,
            axial_repeat_A=14.6,
            pitch_A=4.9,
            handedness="left",
        )
