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
    pytest.mark.skipif(
        any(shutil.which(tool) is None for tool in ("antechamber", "parmchk2", "tleap")),
        reason="AmberTools fragment tools are not available",
    ),
]


def test_pipeline_ordering_enabled_writes_summary(tmp_path: Path) -> None:
    outdir = tmp_path / "ordered_out"
    overrides = (
        "topology.backbone.dp=2 "
        "topology.selector.enabled=true topology.selector.sites=[C6] "
        "ordering.enabled=true ordering.max_candidates=4 "
        "ordering.soft_n_stages=1 ordering.soft_max_iterations=5 ordering.full_max_iterations=5 "
        "forcefield/options=runtime output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=_ROOT, env=env
    )

    report_path = outdir / "build_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["ordering_enabled"] is True
    assert isinstance(data["ordering_summary"], dict)
    assert data["ordering_summary"]["objective"] == "negative_stage2_energy_kj_mol"
    assert data["ordering_summary"]["stage1_nonbonded_mode"] == "soft"
    assert data["ordering_summary"]["stage2_nonbonded_mode"] == "full"
    assert "final_hbond_geometric_fraction" in data["ordering_summary"]
    assert "final_hbond_like_donor_occupancy_fraction" in data["ordering_summary"]
    assert "final_selector_aromatic_stacking_A" in data["ordering_summary"]
    assert "qc_hbond_like_donor_occupancy_fraction" in data
    assert "qc_selector_aromatic_ring_planarity_A" in data
    assert "qc_selector_aromatic_stacking_A" in data
    assert "min_centroid_distance_A" in data["qc_selector_aromatic_stacking_A"]
    assert "max_selector_aromatic_ring_max_deviation_A" in data["qc_thresholds"]


def test_pipeline_runtime_relax_skip_full_stage_writes_soft_stage_summary(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "runtime_relax_skip_full"
    overrides = (
        "topology.backbone.dp=1 "
        "topology.selector.enabled=true topology.selector.sites=[C6] "
        "forcefield/options=runtime_relax "
        "forcefield.options.soft_n_stages=1 "
        "forcefield.options.soft_max_iterations=5 "
        "forcefield.options.full_max_iterations=5 "
        "forcefield.options.hbond_k=0.0 "
        "forcefield.options.skip_full_stage=true "
        "forcefield.options.anneal.enabled=false "
        "output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=_ROOT, env=env
    )

    report_path = outdir / "build_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["relax_enabled"] is True
    assert data["relax_mode"] == "two_stage_runtime"
    assert data["relax_summary"]["protocol"] == "two_stage_runtime"
    assert data["relax_summary"]["protocol_summary"]["skip_full_stage"] is True
    assert data["relax_summary"]["full_stage_skipped"] is True
    assert data["relax_summary"]["stage1_nonbonded_mode"] == "soft"
    assert data["relax_summary"]["stage2_nonbonded_mode"] is None
    assert data["relax_summary"]["final_stage_nonbonded_mode"] == "soft"
