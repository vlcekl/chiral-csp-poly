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
pytestmark = pytest.mark.integration


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
        "output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["forcefield_enabled"] is True
    assert report["forcefield_mode"] == "runtime"
    assert report["forcefield_summary"]["nonbonded_mode"] == "full"
    assert report["forcefield_summary"]["particle_count"] > 0
    assert report["forcefield_summary"]["bonded_term_summary"]["bonds"] > 0
    assert report["forcefield_summary"]["force_inventory"]["counts"]["NonbondedForce"] == 1
    assert report["forcefield_summary"]["exception_summary"]["expected_14_pairs"] > 0
    assert report["forcefield_summary"]["exception_summary"]["expected_14_pairs"] == report["forcefield_summary"]["exception_summary"]["found_14_pairs"]
    assert report["relax_enabled"] is False


def test_pipeline_runtime_cellulose(tmp_path: Path) -> None:
    outdir = tmp_path / "cellulose_runtime"
    _run_build(
        "topology/backbone=cellulose "
        "structure/helix=cellulose_i "
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield/options=runtime "
        "output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["polymer"] == "cellulose"
    assert report["forcefield_mode"] == "runtime"
    assert report["forcefield_summary"]["particle_count"] > 0
    assert report["forcefield_summary"]["force_inventory"]["counts"]["NonbondedForce"] == 1
    assert report["forcefield_summary"]["exception_summary"]["counts_by_rule_bucket"]["backbone_backbone"] > 0


def test_pipeline_runtime_writes_pdbqt_and_amber_exports(tmp_path: Path) -> None:
    outdir = tmp_path / "runtime_exports"
    _run_build(
        "topology.backbone.dp=2 "
        "topology.selector.enabled=false "
        "forcefield/options=runtime "
        "output.export_formats=[pdb,pdbqt,amber] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["amber_enabled"] is True
    assert report["amber_summary"]["parameter_backend"] == "runtime_system_export"
    assert report["docking_enabled"] is True
    assert report["docking_summary"]["backend"] == "native_openmm_charge_export"
    assert (outdir / "receptor.pdbqt").exists()
    assert (outdir / "vina_box.txt").exists()
    assert (outdir / "model.prmtop").exists()
    assert (outdir / "model.inpcrd").exists()


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("antechamber", "parmchk2", "tleap")),
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
        "forcefield.options.soft_n_stages=1 "
        "forcefield.options.soft_max_iterations=10 "
        "forcefield.options.full_max_iterations=10 "
        "forcefield.options.final_restraint_factor=0.2 "
        "forcefield.options.anneal.n_steps=100 "
        "output.export_formats=[pdb,sdf] "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["forcefield_mode"] == "runtime"
    assert report["relax_enabled"] is True
    assert report["relax_mode"] == "two_stage_runtime"
    assert report["relax_summary"]["protocol"] == "two_stage_runtime"
    assert report["relax_summary"]["protocol_summary"]["soft_n_stages"] == 1
    assert report["relax_summary"]["protocol_summary"]["soft_max_iterations"] == 10
    assert report["relax_summary"]["protocol_summary"]["full_max_iterations"] == 10
    assert report["relax_summary"]["restraint_summary"]["freeze_backbone"] is True
    assert report["relax_summary"]["n_selector_atoms"] > 0
    assert report["relax_summary"]["n_connector_atoms"] > 0


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("antechamber", "parmchk2", "tleap")),
    reason="AmberTools fragment tools are not available",
)
@pytest.mark.skipif(
    os.environ.get("POLYCSP_RUN_SLOW") != "1",
    reason="set POLYCSP_RUN_SLOW=1 to run selector-bearing runtime integration",
)
def test_pipeline_runtime_multi_opt_writes_ranked_export_bundles(tmp_path: Path) -> None:
    outdir = tmp_path / "runtime_multi_opt_exports"
    _run_build(
        "topology.backbone.dp=1 "
        "topology.selector.enabled=true "
        "topology.selector.sites=[C6] "
        "forcefield/options=runtime "
        "ordering.enabled=true "
        "ordering.max_candidates=4 "
        "ordering.soft_n_stages=1 "
        "ordering.soft_max_iterations=5 "
        "ordering.full_max_iterations=5 "
        "multi_opt.enabled=true "
        "multi_opt.n_starts=2 "
        "multi_opt.top_k=2 "
        "multi_opt.n_workers=1 "
        "multi_opt.seed=7 "
        "output.export_formats=[pdb,pdbqt,amber] "
        f"output.dir={outdir}"
    )

    rank1 = outdir / "ranked_001"
    rank2 = outdir / "ranked_002"
    for rank_dir in (rank1, rank2):
        assert (rank_dir / "receptor.pdbqt").exists()
        assert (rank_dir / "vina_box.txt").exists()
        assert (rank_dir / "model.prmtop").exists()
        assert (rank_dir / "model.inpcrd").exists()


def test_pipeline_runtime_rejects_periodic_mode(tmp_path: Path) -> None:
    outdir = tmp_path / "periodic_reject"
    with pytest.raises(subprocess.CalledProcessError):
        _run_build(
            "topology/backbone=amylose_periodic "
            "topology.selector.enabled=false "
            "forcefield/options=runtime "
            "output.export_formats=[pdb,sdf] "
            f"output.dir={outdir}"
        )
