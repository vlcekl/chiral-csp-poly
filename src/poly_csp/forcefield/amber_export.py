"""AMBER export helpers built on the canonical runtime model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from rdkit import Chem

from poly_csp.forcefield.export_bundle import ExportBundle
from poly_csp.forcefield.gaff import build_fragment_prmtop, parameterize_gaff_fragment
from poly_csp.forcefield.glycam import _ensure_required_tools, _run_command
from poly_csp.io.pdb import write_pdb_from_rdkit


def parameterize_selector_fragment(
    selector_mol: Chem.Mol,
    charge_model: str = "bcc",
    net_charge: int = 0,
    work_dir: Path | None = None,
) -> Dict[str, str]:
    """Run Antechamber/Parmchk2 on a selector fragment via GAFF helpers."""
    return parameterize_gaff_fragment(
        fragment_mol=selector_mol,
        charge_model=charge_model,
        net_charge=net_charge,
        residue_name="SEL",
        pdb_name="selector.pdb",
        mol2_name="selector.mol2",
        frcmod_name="selector.frcmod",
        lib_name="selector.lib",
        work_dir=work_dir,
        ensure_tools_fn=_ensure_required_tools,
        run_command_fn=_run_command,
        write_pdb_fn=write_pdb_from_rdkit,
    )


def build_selector_prmtop(
    mol2_path: str | Path,
    frcmod_path: str | Path,
    work_dir: Path | None = None,
) -> str:
    """Create a standalone AMBER prmtop for the selector fragment."""
    return build_fragment_prmtop(
        mol2_path=mol2_path,
        frcmod_path=frcmod_path,
        prmtop_name="selector.prmtop",
        inpcrd_name="selector.inpcrd",
        clean_mol2_name="selector_clean.mol2",
        work_dir=work_dir,
        run_command_fn=_run_command,
    )


def _load_parmed():
    try:
        from parmed import openmm as pmd_openmm
    except ImportError as exc:  # pragma: no cover - exercised in environments without ParmEd
        raise RuntimeError(
            "AMBER export requires ParmEd to be installed in the runtime environment."
        ) from exc
    return pmd_openmm


def export_amber_artifacts(
    bundle: ExportBundle,
    outdir: str | Path,
    model_name: str = "model",
) -> Dict[str, object]:
    """Export AMBER artifacts directly from the canonical runtime system."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    pmd_openmm = _load_parmed()
    structure = pmd_openmm.load_topology(
        bundle.topology,
        system=bundle.system_build.system,
        xyz=bundle.positions_nm,
        box=bundle.box_vectors_nm,
    )

    prmtop_path = out / f"{model_name}.prmtop"
    inpcrd_path = out / f"{model_name}.inpcrd"
    structure.save(str(prmtop_path), overwrite=True)
    structure.save(str(inpcrd_path), overwrite=True)

    total_charge_e = sum(particle.charge_e for particle in bundle.nonbonded_particles)
    summary: Dict[str, object] = {
        "enabled": True,
        "parameter_backend": "runtime_system_export",
        "source_manifest": dict(bundle.system_build.source_manifest),
        "particle_count": int(bundle.mol.GetNumAtoms()),
        "total_charge_e": float(total_charge_e),
        "files": {
            "prmtop": str(prmtop_path),
            "inpcrd": str(inpcrd_path),
        },
    }
    if bundle.box_vectors_nm is not None:
        summary["box_vectors_nm"] = [
            [float(vec[0]), float(vec[1]), float(vec[2])]
            for vec in bundle.box_vectors_nm
        ]

    manifest_path = out / "amber_export.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    return summary
