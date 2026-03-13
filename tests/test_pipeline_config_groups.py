from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from poly_csp.cache_versions import (
    BACKBONE_POSE_CACHE_SCHEMA_VERSION,
    BACKBONE_POSE_MODEL_VERSION,
)


_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def _run_build(overrides: str) -> None:
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=_ROOT, env=env
    )


@pytest.mark.parametrize(
    ("backbone_preset", "extra_overrides", "polymer", "dp", "helix_name", "axial_repeat_A"),
    [
        ("amylose_periodic", "", "amylose", 4, "amylose_CSP_4_3_derivatized", 14.6),
        (
            "cellulose_periodic",
            "structure/helix=cellulose_3_2_derivatized ",
            "cellulose",
            3,
            "cellulose_CSP_3_2_derivatized",
            16.2,
        ),
    ],
)
def test_pipeline_topology_backbone_group_override_runs(
    tmp_path: Path,
    backbone_preset: str,
    extra_overrides: str,
    polymer: str,
    dp: int,
    helix_name: str,
    axial_repeat_A: float,
) -> None:
    outdir = tmp_path / f"{backbone_preset}_out"
    _run_build(
        f"topology/backbone={backbone_preset} "
        f"{extra_overrides}"
        "topology.selector.enabled=false "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == polymer
    assert report["dp"] == dp
    assert report["end_mode"] == "periodic"
    assert report["helix_name"] == helix_name
    assert report["axial_repeat_A"] == pytest.approx(axial_repeat_A)
    assert report["backbone_pose_cache"]["kind"] in {"build", "disk"}
    assert report["backbone_pose_cache"]["entry_path"].endswith("pose.json")
    assert report["backbone_pose_cache"]["cache_dir"].endswith(
        ".cache/poly_csp/backbone_pose"
    )
    assert report["backbone_pose_cache"]["schema_version"] == BACKBONE_POSE_CACHE_SCHEMA_VERSION
    assert report["backbone_pose_cache"]["model_version"] == BACKBONE_POSE_MODEL_VERSION


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


def test_pipeline_selector_group_override_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "selector_group_out"
    _run_build(
        "topology/selector=35dcpc "
        "topology.backbone.dp=1 "
        "ordering.enabled=false "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["selector_enabled"] is True
    assert report["selector_name"] == "35dcpc"
    assert report["selector_sites"] == ["C2", "C3", "C6"]


def test_pipeline_phase_group_override_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "phase_group_out"
    _run_build(
        "phase=chiralcel_oz "
        "topology.backbone.dp=1 "
        "ordering.enabled=false "
        "forcefield.options.enabled=false output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == "cellulose"
    assert report["selector_name"] == "3c4mpc"
    assert report["helix_name"] == "cellulose_CSP_3_2_derivatized"
    assert report["phase_column_id"] == "OZ"
    assert report["phase_name"] == "Chiralcel OZ"
    assert report["phase_attachment_mode"] == "coated"


def test_pipeline_phase_group_periodic_override_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "phase_group_periodic_out"
    _run_build(
        "phase=chiralcel_oz "
        "topology.backbone.end_mode=periodic "
        "topology.backbone.dp=3 "
        "ordering.enabled=false "
        "forcefield/options=runtime "
        "output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == "cellulose"
    assert report["end_mode"] == "periodic"
    assert report["phase_column_id"] == "OZ"
    assert report["phase_name"] == "Chiralcel OZ"
    assert report["periodic_box_A"] is not None
    assert report["forcefield_summary"]["exception_summary"]["periodic"] is True


def test_pipeline_ordering_solvent_ready_group_override_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "ordering_solvent_ready_out"
    _run_build(
        "topology.backbone.dp=2 "
        "topology.selector.enabled=true "
        "topology.selector.sites=[C6] "
        "ordering=solvent_ready "
        "ordering.max_candidates=4 "
        "ordering.soft_n_stages=1 "
        "ordering.soft_max_iterations=5 "
        "ordering.full_max_iterations=5 "
        "forcefield/options=runtime_seed "
        "output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["ordering_enabled"] is True
    assert report["ordering_summary"]["objective"] == "negative_stage1_energy_kj_mol"
    assert report["ordering_summary"]["full_stage_skipped"] is True
    assert report["ordering_summary"]["final_stage_nonbonded_mode"] == "soft"
    assert report["ordering_summary"]["stage2_nonbonded_mode"] is None
    assert report["ordering_summary"]["final_selector_aromatic_stacking_A"]
    assert report["forcefield_mode"] == "runtime"
    assert report["relax_enabled"] is False


def test_pipeline_forcefield_runtime_seed_relax_group_override_runs(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "runtime_seed_relax_out"
    _run_build(
        "topology.backbone.dp=1 "
        "topology.selector.enabled=true "
        "topology.selector.sites=[C6] "
        "ordering.enabled=false "
        "forcefield/options=runtime_seed_relax "
        "forcefield.options.soft_n_stages=1 "
        "forcefield.options.soft_max_iterations=5 "
        "forcefield.options.full_max_iterations=5 "
        "output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["forcefield_mode"] == "runtime"
    assert report["relax_enabled"] is True
    assert report["relax_summary"]["protocol"] == "two_stage_runtime"
    assert report["relax_summary"]["protocol_summary"]["skip_full_stage"] is True
    assert report["relax_summary"]["full_stage_skipped"] is True
    assert report["relax_summary"]["stage1_nonbonded_mode"] == "soft"
    assert report["relax_summary"]["stage2_nonbonded_mode"] is None
    assert report["relax_summary"]["final_stage_nonbonded_mode"] == "soft"


def test_ranked_periodic_cell_subreport_uses_main_backbone_pose_cache_metadata() -> None:
    report = {
        "backbone_pose_cache": {
            "kind": "disk",
            "cache_dir": ".cache/poly_csp/backbone_pose",
            "entry_path": ".cache/poly_csp/backbone_pose/example/pose.json",
        }
    }

    rank_periodic_cell_report = {
        "rank": 1,
        "score": 0.0,
        "seed_used": 7,
        "end_mode": "periodic",
        "output_end_mode": "periodic",
        "backbone_pose_cache": report["backbone_pose_cache"],
    }

    assert rank_periodic_cell_report["backbone_pose_cache"] == report["backbone_pose_cache"]
