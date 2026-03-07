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


def _run_build(overrides: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    return subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=_ROOT, env=env
    )


def test_pipeline_relax_disabled_writes_summary(tmp_path: Path) -> None:
    """Verify that relax_enabled=false is recorded in the build report."""
    outdir = tmp_path / "relax_out"
    _run_build(
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield.options.enabled=false "
        "amber.enabled=false "
        f"output.dir={outdir}"
    )

    report_path = outdir / "build_report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["relax_enabled"] is False


def test_pipeline_qc_production_hard_fails(tmp_path: Path) -> None:
    outdir = tmp_path / "qc_prod_out"
    with pytest.raises(subprocess.CalledProcessError):
        _run_build(
            "topology.backbone.dp=2 topology.selector.enabled=false qc=production "
            "forcefield.options.enabled=false amber.enabled=false "
            "qc.min_heavy_distance_A=10.0 "
            f"output.dir={outdir}"
        )
