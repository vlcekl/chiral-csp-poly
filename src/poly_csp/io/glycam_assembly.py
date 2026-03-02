# poly_csp/io/glycam_assembly.py
"""Residue-aware tleap assembly using GLYCAM06 + GAFF2.

Builds the AMBER topology by:
1. Loading GLYCAM06j for polysaccharide backbone residues.
2. Loading GAFF2 for selector fragments.
3. Assembling the system residue-by-residue with proper linkage records.

This replaces the monolithic antechamber approach from the original
``ambertools`` backend and ensures:
- Every glucose unit has identical GLYCAM06-derived parameters.
- Every selector has identical GAFF2-derived parameters.
- Charges are derived once per fragment and replicated.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

from rdkit import Chem

from poly_csp.io.pdb import write_pdb_from_rdkit


# GLYCAM06j residue codes for glucose units in 1->4 linked chains.
# Convention: first char = linkage to next residue, rest = sugar code.
# See GLYCAM06 parameter documentation.
GLYCAM_RESIDUE_NAMES = {
    ("amylose", "terminal_nonreducing"): "0GA",   # α-D-Glc, no downstream linkage
    ("amylose", "internal"):             "4GA",   # α-D-Glc, 1->4 linked
    ("amylose", "terminal_reducing"):    "4GA",   # reducing end (same type; capped separately)
    ("cellulose", "terminal_nonreducing"): "0GB",  # β-D-Glc
    ("cellulose", "internal"):             "4GB",  # β-D-Glc, 1->4 linked
    ("cellulose", "terminal_reducing"):    "4GB",
}


def _ensure_required_tools(tools: Sequence[str]) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            "GLYCAM assembly requires executables not found on PATH: "
            + ", ".join(missing)
            + ". Install AmberTools with GLYCAM06 support."
        )


def _run_command(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    proc = subprocess.run(
        list(cmd), cwd=str(cwd), text=True, capture_output=True, check=False,
    )
    combined = (
        f"$ {' '.join(cmd)}\n\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}. "
            f"See: {log_path}"
        )


def build_glycam_sequence(
    polymer: str,
    dp: int,
) -> List[str]:
    """Build a sequence of GLYCAM residue codes for a linear 1->4 chain.

    Returns a list of residue names, one per glucose unit:
    - First residue: terminal (non-reducing end)
    - Middle residues: internal
    - Last residue: terminal (reducing end)
    """
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}")

    key_base = polymer.lower()
    if dp == 1:
        return [GLYCAM_RESIDUE_NAMES[(key_base, "terminal_nonreducing")]]

    seq: List[str] = []
    seq.append(GLYCAM_RESIDUE_NAMES[(key_base, "terminal_nonreducing")])
    for _ in range(dp - 2):
        seq.append(GLYCAM_RESIDUE_NAMES[(key_base, "internal")])
    seq.append(GLYCAM_RESIDUE_NAMES[(key_base, "terminal_reducing")])
    return seq


def build_tleap_script(
    polymer: str,
    dp: int,
    selector_lib_path: str | None = None,
    selector_frcmod_path: str | None = None,
    model_name: str = "model",
    prmtop_name: str = "model.prmtop",
    inpcrd_name: str = "model.inpcrd",
) -> str:
    """Generate a tleap input script for residue-aware assembly.

    The script:
    1. Sources GLYCAM06j for backbone residues.
    2. Optionally loads GAFF2 selector library/frcmod.
    3. Assembles the polymer chain from residue codes.
    4. Saves prmtop/inpcrd.
    """
    lines: List[str] = [
        "# poly_csp residue-aware GLYCAM06 + GAFF2 assembly",
        "source leaprc.GLYCAM_06j-1",
    ]

    if selector_lib_path or selector_frcmod_path:
        lines.append("source leaprc.gaff2")
    if selector_frcmod_path:
        lines.append(f"loadamberparams {selector_frcmod_path}")
    if selector_lib_path:
        lines.append(f"loadoff {selector_lib_path}")

    seq = build_glycam_sequence(polymer, dp)
    seq_str = " ".join(f"{{ {res} }}" for res in seq)
    lines.append(f"mol = sequence {{ {seq_str} }}")

    lines.extend([
        f"saveamberparm mol {prmtop_name} {inpcrd_name}",
        "quit",
    ])
    return "\n".join(lines) + "\n"


def run_tleap_assembly(
    tleap_script: str,
    outdir: Path,
    model_name: str = "model",
) -> Dict[str, object]:
    """Execute tleap with the given script and return artifact paths.

    Parameters
    ----------
    tleap_script : str
        Contents of the tleap input file.
    outdir : Path
        Directory for all output files.
    model_name : str
        Base name for prmtop/inpcrd.

    Returns
    -------
    Dict with keys: prmtop, inpcrd, tleap_input, tleap_log, parameterized.
    """
    _ensure_required_tools(("tleap",))

    outdir.mkdir(parents=True, exist_ok=True)
    tleap_path = outdir / "tleap.in"
    tleap_log = outdir / "tleap.log"
    prmtop_path = outdir / f"{model_name}.prmtop"
    inpcrd_path = outdir / f"{model_name}.inpcrd"

    tleap_path.write_text(tleap_script, encoding="utf-8")
    _run_command(["tleap", "-f", tleap_path.name], cwd=outdir, log_path=tleap_log)

    missing = [
        str(p) for p in (prmtop_path, inpcrd_path)
        if not p.exists() or p.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "tleap assembly completed but expected outputs were not generated: "
            + ", ".join(missing) + f". See: {tleap_log}"
        )

    return {
        "prmtop": str(prmtop_path),
        "inpcrd": str(inpcrd_path),
        "tleap_input": str(tleap_path),
        "tleap_log": str(tleap_log),
        "parameterized": True,
        "parameter_backend": "residue_aware",
    }


def parameterize_selector_fragment(
    selector_mol: Chem.Mol,
    charge_model: str = "bcc",
    net_charge: int = 0,
    work_dir: Path | None = None,
) -> Dict[str, str]:
    """Run antechamber + parmchk2 on a single selector fragment.

    Returns dict with paths to mol2, frcmod, and lib files.
    """
    _ensure_required_tools(("antechamber", "parmchk2"))

    import tempfile
    if work_dir is None:
        wd = Path(tempfile.mkdtemp(prefix="polycsp_sel_"))
    else:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)

    pdb_path = wd / "selector.pdb"
    mol2_path = wd / "selector.mol2"
    frcmod_path = wd / "selector.frcmod"
    lib_path = wd / "selector.lib"

    write_pdb_from_rdkit(selector_mol, pdb_path)

    # antechamber
    _run_command(
        [
            "antechamber",
            "-i", pdb_path.name, "-fi", "pdb",
            "-o", mol2_path.name, "-fo", "mol2",
            "-at", "gaff2", "-c", charge_model,
            "-nc", str(net_charge),
            "-rn", "SEL", "-dr", "no", "-pf", "y", "-s", "2",
        ],
        cwd=wd,
        log_path=wd / "antechamber.log",
    )

    # parmchk2
    _run_command(
        [
            "parmchk2",
            "-i", mol2_path.name, "-f", "mol2",
            "-s", "gaff2", "-o", frcmod_path.name,
        ],
        cwd=wd,
        log_path=wd / "parmchk2.log",
    )

    # saveoff via tleap
    saveoff_script = "\n".join([
        "source leaprc.gaff2",
        f"loadamberparams {frcmod_path.name}",
        f"sel = loadmol2 {mol2_path.name}",
        f"saveoff sel {lib_path.name}",
        "quit",
    ]) + "\n"
    saveoff_in = wd / "saveoff.in"
    saveoff_in.write_text(saveoff_script, encoding="utf-8")
    _run_command(
        ["tleap", "-f", saveoff_in.name],
        cwd=wd,
        log_path=wd / "saveoff.log",
    )

    return {
        "mol2": str(mol2_path),
        "frcmod": str(frcmod_path),
        "lib": str(lib_path),
    }
