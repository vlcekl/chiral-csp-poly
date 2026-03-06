from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("tleap") is None, reason="AmberTools/tleap not available"),
]


def _run_build(overrides: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        cwd=_ROOT,
        env=env,
    )


def test_pipeline_backbone_glycam_only_amylose(tmp_path: Path) -> None:
    outdir = tmp_path / "amylose_glycam"
    _run_build(
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield/options=backbone_glycam_only "
        "amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["forcefield_enabled"] is True
    assert report["forcefield_mode"] == "backbone_glycam_only"
    assert report["forcefield_summary"]["nonbonded_mode"] == "glycam_no_cutoff"
    assert report["forcefield_summary"]["particle_count"] > 0


def test_pipeline_backbone_glycam_only_cellulose(tmp_path: Path) -> None:
    outdir = tmp_path / "cellulose_glycam"
    _run_build(
        "topology/backbone=cellulose "
        "structure/helix=cellulose_i "
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield/options=backbone_glycam_only "
        "amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == "cellulose"
    assert report["forcefield_mode"] == "backbone_glycam_only"
    assert report["forcefield_summary"]["particle_count"] > 0


def test_pipeline_backbone_glycam_only_rejects_selectors(tmp_path: Path) -> None:
    outdir = tmp_path / "selector_reject"
    with pytest.raises(subprocess.CalledProcessError):
        _run_build(
            "topology.backbone.dp=2 "
            "topology.selector.enabled=true "
            "forcefield/options=backbone_glycam_only "
            "amber.enabled=false "
            f"output.dir={outdir}"
        )


def test_pipeline_backbone_glycam_only_rejects_periodic_mode(tmp_path: Path) -> None:
    outdir = tmp_path / "periodic_reject"
    with pytest.raises(subprocess.CalledProcessError):
        _run_build(
            "topology/backbone=amylose_periodic "
            "topology.selector.enabled=false "
            "forcefield/options=backbone_glycam_only "
            "amber.enabled=false "
            f"output.dir={outdir}"
        )

