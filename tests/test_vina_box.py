from __future__ import annotations

from pathlib import Path

from poly_csp.io.vina import VinaBoxSpec, build_vina_box, write_vina_box
from poly_csp.topology.selectors import SelectorRegistry
from tests.support import build_forcefield_mol, test_helix as make_test_helix


def test_build_vina_box_uses_central_residue_window_and_writes_cli_args(
    tmp_path: Path,
) -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(dp=6, selector=selector, site="C6")
    box = build_vina_box(
        mol,
        helix=make_test_helix(),
        spec=VinaBoxSpec(buffer_A=6.0),
    )

    assert box.residue_indices == (1, 2, 3, 4)
    assert box.heavy_atom_count > 0
    assert box.size_x > 0.0
    assert box.size_y > 0.0
    assert box.size_z > 0.0

    summary = write_vina_box(box, tmp_path / "vina_box.txt")
    text = Path(summary["file"]).read_text(encoding="utf-8")
    assert "--center_x" in text
    assert "--center_y" in text
    assert "--center_z" in text
    assert "--size_x" in text
    assert "--size_y" in text
    assert "--size_z" in text
