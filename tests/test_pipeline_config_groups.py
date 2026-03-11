from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def _run_build(overrides: str) -> None:
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=_ROOT, env=env
    )


def test_pipeline_topology_backbone_group_override_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "periodic_backbone_out"
    _run_build(
        "topology/backbone=amylose_periodic "
        "topology.selector.enabled=false "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == "amylose"
    assert report["dp"] == 4
    assert report["end_mode"] == "periodic"
    assert report["helix_name"] == "amylose_CSP_4_3_derivatized"
    assert report["axial_repeat_A"] == pytest.approx(14.6)


def test_pipeline_structure_helix_group_override_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "cellulose_helix_out"
    _run_build(
        "topology/backbone=cellulose "
        "structure/helix=cellulose_3_2_derivatized "
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == "cellulose"
    assert report["helix_name"] == "cellulose_CSP_3_2_derivatized"
    assert report["axial_repeat_A"] == pytest.approx(16.2)
    assert report["qc_pass"] is True
