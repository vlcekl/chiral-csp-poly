# tests/test_glycam_assembly.py
"""Unit tests for GLYCAM assembly logic (no AmberTools dependency)."""
from __future__ import annotations

import pytest

from poly_csp.io.glycam_assembly import (
    build_glycam_sequence,
    build_tleap_script,
)


def test_glycam_sequence_amylose_dp4() -> None:
    seq = build_glycam_sequence("amylose", dp=4)
    assert len(seq) == 4
    assert seq[0] == "0GA"
    assert seq[-1] == "4GA"
    assert all(r == "4GA" for r in seq[1:])


def test_glycam_sequence_cellulose_dp2() -> None:
    seq = build_glycam_sequence("cellulose", dp=2)
    assert len(seq) == 2
    assert seq[0] == "0GB"
    assert seq[1] == "4GB"


def test_glycam_sequence_dp1() -> None:
    seq = build_glycam_sequence("amylose", dp=1)
    assert seq == ["0GA"]


def test_glycam_sequence_rejects_dp0() -> None:
    with pytest.raises(ValueError, match="dp"):
        build_glycam_sequence("amylose", dp=0)


def test_tleap_script_backbone_only() -> None:
    script = build_tleap_script(polymer="amylose", dp=4)
    assert "GLYCAM_06j" in script
    assert "0GA" in script
    assert "4GA" in script
    assert "saveamberparm" in script
    # Should NOT load GAFF2 leaprc if no selectors
    assert "leaprc.gaff2" not in script


def test_tleap_script_with_selector() -> None:
    script = build_tleap_script(
        polymer="amylose",
        dp=4,
        selector_lib_path="/tmp/sel.lib",
        selector_frcmod_path="/tmp/sel.frcmod",
    )
    assert "GLYCAM_06j" in script
    assert "gaff2" in script
    assert "/tmp/sel.lib" in script
    assert "/tmp/sel.frcmod" in script
    # Should have both GLYCAM and GAFF2
    assert "0GA" in script
