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


def test_pipeline_runtime_amylose(tmp_path: Path) -> None:
    outdir = tmp_path / "amylose_runtime"
    _run_build(
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield/options=runtime "
        "amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["forcefield_enabled"] is True
    assert report["forcefield_mode"] == "runtime"
    assert report["forcefield_summary"]["nonbonded_mode"] == "full"
    assert report["forcefield_summary"]["particle_count"] > 0
    assert report["relax_enabled"] is False


def test_pipeline_runtime_cellulose(tmp_path: Path) -> None:
    outdir = tmp_path / "cellulose_runtime"
    _run_build(
        "topology/backbone=cellulose "
        "structure/helix=cellulose_i "
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield/options=runtime "
        "amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == "cellulose"
    assert report["forcefield_mode"] == "runtime"
    assert report["forcefield_summary"]["particle_count"] > 0


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("antechamber", "parmchk2")),
    reason="AmberTools fragment tools are not available",
)
@pytest.mark.skipif(
    os.environ.get("POLYCSP_RUN_SLOW") != "1",
    reason="set POLYCSP_RUN_SLOW=1 to run selector-bearing runtime integration",
)
def test_pipeline_runtime_relax_selector_system(tmp_path: Path) -> None:
    outdir = tmp_path / "selector_runtime_relax"
    _run_build(
        "topology.backbone.dp=1 "
        "topology.selector.enabled=true "
        "topology.selector.sites=[C6] "
        "forcefield/options=runtime_relax "
        "forcefield.options.n_stages=1 "
        "forcefield.options.max_iterations=10 "
        "forcefield.options.anneal.n_steps=100 "
        "amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["forcefield_mode"] == "runtime"
    assert report["relax_enabled"] is True
    assert report["relax_mode"] == "two_stage_runtime"
    assert report["relax_summary"]["protocol"] == "two_stage_runtime"
    assert report["relax_summary"]["n_selector_atoms"] > 0
    assert report["relax_summary"]["n_connector_atoms"] > 0


def test_pipeline_runtime_rejects_periodic_mode(tmp_path: Path) -> None:
    outdir = tmp_path / "periodic_reject"
    with pytest.raises(subprocess.CalledProcessError):
        _run_build(
            "topology/backbone=amylose_periodic "
            "topology.selector.enabled=false "
            "forcefield/options=runtime "
            "amber.enabled=false "
            f"output.dir={outdir}"
        )
