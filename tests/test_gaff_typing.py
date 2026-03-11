from __future__ import annotations

import shutil

import pytest

from poly_csp.forcefield.gaff import load_selector_fragment_params
from poly_csp.topology.selectors import SelectorRegistry


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("antechamber", "parmchk2", "tleap")),
    reason="AmberTools executables not available",
)
def test_selector_gaff_typing_preserves_aromatic_ring(tmp_path) -> None:
    selector = SelectorRegistry.get("35dmpc")
    params = load_selector_fragment_params(selector, work_dir=tmp_path)

    mol2_text = (tmp_path / "selector.mol2").read_text(encoding="utf-8")
    assert " ca " in mol2_text
    assert " ar " in mol2_text

    short_ring_bonds = [
        bond
        for bond in params.bonds
        if bond.atom_names[0].startswith("S")
        and bond.atom_names[1].startswith("S")
        and bond.length_nm < 0.147
    ]
    assert len(short_ring_bonds) >= 4
