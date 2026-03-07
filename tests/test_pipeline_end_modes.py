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


def test_pipeline_capped_mode_runs_with_explicit_caps(tmp_path: Path) -> None:
    outdir = tmp_path / "capped_out"
    _run_build(
        "topology.backbone.dp=3 topology.selector.enabled=false "
        "topology.backbone.end_mode=capped "
        "+topology.backbone.end_caps.left=acetyl "
        "+topology.backbone.end_caps.right=methoxy "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["end_mode"] == "capped"
    assert report["qc_pass"] is True
    assert (outdir / "model.pdb").exists()


def test_pipeline_periodic_natural_oh_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "periodic_out"
    _run_build(
        "topology.backbone.dp=4 topology.selector.enabled=false "
        "topology.backbone.monomer_representation=natural_oh "
        "topology.backbone.end_mode=periodic "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["end_mode"] == "periodic"
    assert report["monomer_representation"] == "natural_oh"
    assert report["qc_pass"] is True
    assert (outdir / "model.pdb").exists()


def test_pipeline_writes_all_atom_model_by_default(tmp_path: Path) -> None:
    outdir = tmp_path / "all_atom_out"
    _run_build(
        "topology.backbone.dp=1 topology.selector.enabled=false "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["all_atom_atom_count"] is not None
    assert report["all_atom_backbone_h_count"] is not None
    assert report["all_atom_manifest_schema_version"] == 1

    model_pdb = outdir / "model.pdb"
    assert model_pdb.exists()
    assert not (outdir / "model_heavy.pdb").exists()
