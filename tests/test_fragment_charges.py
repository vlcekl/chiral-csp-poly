# tests/test_fragment_charges.py
"""Unit tests for fragment charge derivation logic."""
from __future__ import annotations

import pytest
from poly_csp.io.fragment_charges import (
    neutralize_charges,
    replicate_charges,
)


def test_replicate_charges_repeats_exactly() -> None:
    template = [0.1, -0.2, 0.3, -0.2]
    result = replicate_charges(template, 4)
    assert len(result) == 16
    for i in range(4):
        assert result[i * 4 : (i + 1) * 4] == template


def test_replicate_charges_single_copy() -> None:
    template = [0.5, -0.5]
    assert replicate_charges(template, 1) == template


def test_replicate_charges_rejects_zero() -> None:
    with pytest.raises(ValueError, match="n_copies"):
        replicate_charges([0.1], 0)


def test_neutralize_charges_adjusts_to_target() -> None:
    charges = [0.1, 0.2, 0.3, 0.4]
    result = neutralize_charges(charges, target_charge=0)
    assert abs(sum(result)) < 1e-12


def test_neutralize_charges_preserves_relative_differences() -> None:
    charges = [0.5, -0.3, 0.2]
    result = neutralize_charges(charges, target_charge=0)
    # Relative differences between atoms must be preserved
    assert abs((result[0] - result[1]) - (charges[0] - charges[1])) < 1e-12
    assert abs((result[1] - result[2]) - (charges[1] - charges[2])) < 1e-12


def test_neutralize_charges_nonzero_target() -> None:
    charges = [0.0, 0.0, 0.0]
    result = neutralize_charges(charges, target_charge=-1)
    assert abs(sum(result) - (-1.0)) < 1e-12
