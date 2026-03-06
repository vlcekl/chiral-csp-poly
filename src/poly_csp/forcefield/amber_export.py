"""AMBER/tleap export helpers built on residue-aware GLYCAM assembly."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from rdkit import Chem

from poly_csp.forcefield.gaff import build_fragment_prmtop, parameterize_gaff_fragment
from poly_csp.forcefield.glycam import (
    _ensure_required_tools,
    _run_command,
    build_glycam_sequence,
    build_linkage_frcmod,
    run_tleap_assembly,
)
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


def _build_export_tleap_script(
    polymer: str,
    dp: int,
    selector_lib_path: str | None = None,
    selector_frcmod_path: str | None = None,
    linkage_frcmod_path: str | None = None,
    model_name: str = "model",
    prmtop_name: str | None = None,
    inpcrd_name: str | None = None,
    periodic: bool = False,
    box_vectors_A: tuple[float, float, float] | None = None,
) -> str:
    prmtop_name = prmtop_name or f"{model_name}.prmtop"
    inpcrd_name = inpcrd_name or f"{model_name}.inpcrd"

    lines = [
        "# poly_csp residue-aware GLYCAM06 + GAFF2 assembly",
        "source leaprc.GLYCAM_06j-1",
    ]
    if selector_lib_path or selector_frcmod_path:
        lines.append("source leaprc.gaff2")
    if selector_frcmod_path:
        lines.append(f"loadamberparams {selector_frcmod_path}")
    if selector_lib_path:
        lines.append(f"loadoff {selector_lib_path}")
    if linkage_frcmod_path:
        lines.append(f"loadamberparams {linkage_frcmod_path}")

    seq_str = " ".join(build_glycam_sequence(polymer, dp))
    lines.append(f"mol = sequence {{ {seq_str} }}")

    if periodic and dp > 1:
        lines.append(f"bond mol.1.C1 mol.{dp}.O4")
    if periodic and box_vectors_A is not None:
        lx, ly, lz = box_vectors_A
        lines.append(f"setBox mol centers {{ {lx:.4f} {ly:.4f} {lz:.4f} }}")

    lines.extend([
        f"saveamberparm mol {prmtop_name} {inpcrd_name}",
        "quit",
    ])
    return "\n".join(lines) + "\n"


def export_amber_artifacts(
    mol: Chem.Mol,
    outdir: str | Path,
    model_name: str = "model",
    charge_model: str = "bcc",
    net_charge: int | str | None = "auto",
    polymer: str = "amylose",
    dp: int | None = None,
    selector_mol: Chem.Mol | None = None,
    periodic: bool = False,
    box_vectors_A: tuple[float, float, float] | None = None,
) -> Dict[str, object]:
    """Export AMBER artifacts using residue-aware GLYCAM + GAFF assembly."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if dp is None:
        if mol.HasProp("_poly_csp_dp"):
            dp = int(mol.GetIntProp("_poly_csp_dp"))
        else:
            raise ValueError(
                "residue-aware backend requires dp (degree of polymerization). "
                "Pass dp= or ensure the molecule has _poly_csp_dp metadata."
            )

    selector_lib_path = None
    selector_frcmod_path = None
    selector_prmtop_path = None
    if selector_mol is not None:
        selector_dir = out / "selector_params"
        resolved_net_charge = 0 if net_charge in {None, "auto"} else int(net_charge)
        selector_artifacts = parameterize_selector_fragment(
            selector_mol=selector_mol,
            charge_model=charge_model,
            net_charge=resolved_net_charge,
            work_dir=selector_dir,
        )
        selector_lib_path = selector_artifacts["lib"]
        selector_frcmod_path = selector_artifacts["frcmod"]
        selector_prmtop_path = build_selector_prmtop(
            mol2_path=selector_artifacts["mol2"],
            frcmod_path=selector_artifacts["frcmod"],
            work_dir=selector_dir,
        )

    linkage_frcmod = build_linkage_frcmod(out)
    pdb_path = out / f"{model_name}.pdb"
    write_pdb_from_rdkit(mol, pdb_path)

    tleap_script = _build_export_tleap_script(
        polymer=polymer,
        dp=dp,
        selector_lib_path=selector_lib_path,
        selector_frcmod_path=selector_frcmod_path,
        linkage_frcmod_path=str(linkage_frcmod.resolve()),
        model_name=model_name,
        prmtop_name=f"{model_name}.prmtop",
        inpcrd_name=f"{model_name}.inpcrd",
        periodic=periodic,
        box_vectors_A=box_vectors_A,
    )
    assembly_result = run_tleap_assembly(
        tleap_script=tleap_script,
        outdir=out,
        model_name=model_name,
    )

    summary: Dict[str, object] = {
        "enabled": True,
        "parameterized": True,
        "charge_model": charge_model,
        "parameter_backend": "residue_aware",
        "polymer": polymer,
        "dp": dp,
        "periodic": bool(periodic),
        "files": {
            "pdb": str(pdb_path),
            "prmtop": assembly_result["prmtop"],
            "inpcrd": assembly_result["inpcrd"],
            "tleap_input": assembly_result["tleap_input"],
            "tleap_log": assembly_result["tleap_log"],
        },
        "notes": [
            "Assembled with GLYCAM06j backbone + GAFF2 selectors.",
            "Charges derived per fragment and replicated for symmetry.",
        ],
    }
    if box_vectors_A is not None:
        summary["box_vectors_A"] = list(box_vectors_A)
    if selector_lib_path:
        summary["files"]["selector_lib"] = selector_lib_path  # type: ignore[index]
        summary["files"]["selector_frcmod"] = selector_frcmod_path  # type: ignore[index]
    if selector_prmtop_path:
        summary["files"]["selector_prmtop"] = selector_prmtop_path  # type: ignore[index]

    manifest_path = out / "amber_export.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    return summary

