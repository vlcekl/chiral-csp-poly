# poly_csp/io/fragment_charges.py
"""Fragment-based partial charge derivation for symmetric polymer parameterization.

Design principle: compute AM1-BCC charges once on a small capped fragment,
then replicate identical charges to every equivalent unit in the polymer.
This ensures every glucose unit carries the same charges and every selector
carries the same charges.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem

from poly_csp.io.pdb import write_pdb_from_rdkit


def _ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Fragment charge derivation requires '{name}' on PATH. "
            "Install AmberTools or set up the correct environment."
        )


def _run_antechamber(
    input_pdb: Path,
    output_mol2: Path,
    charge_model: str,
    net_charge: int,
    cwd: Path,
    log_path: Path,
) -> None:
    cmd = [
        "antechamber",
        "-i", input_pdb.name,
        "-fi", "pdb",
        "-o", output_mol2.name,
        "-fo", "mol2",
        "-at", "gaff2",
        "-c", charge_model,
        "-nc", str(net_charge),
        "-rn", "FRG",
        "-dr", "no",
        "-pf", "y",
        "-s", "2",
    ]
    proc = subprocess.run(
        cmd, cwd=str(cwd), text=True, capture_output=True, check=False,
    )
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"antechamber failed (exit {proc.returncode}). See: {log_path}"
        )


def _parse_mol2_charges(mol2_path: Path) -> List[float]:
    """Extract partial charges from a TRIPOS mol2 file."""
    charges: List[float] = []
    in_atom_block = False
    with open(mol2_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if stripped.startswith("@<TRIPOS>"):
                in_atom_block = False
                continue
            if in_atom_block and stripped:
                parts = stripped.split()
                if len(parts) >= 9:
                    charges.append(float(parts[8]))
    return charges


def _formal_charge(mol: Chem.Mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def derive_fragment_charges(
    mol: Chem.Mol,
    charge_model: str = "bcc",
    net_charge: int | str | None = "auto",
    work_dir: Path | None = None,
) -> List[float]:
    """Derive AM1-BCC partial charges for a single molecular fragment.

    Parameters
    ----------
    mol : RDKit Mol
        The fragment molecule (must have a conformer).
    charge_model : str
        Charge model for antechamber (\"bcc\", \"gas\", \"resp\").
    net_charge : int or \"auto\"
        Net charge; \"auto\" uses RDKit formal charge sum.
    work_dir : Path, optional
        Working directory; uses a temp directory if None.

    Returns
    -------
    List[float]
        Partial charges, one per atom in the fragment.
    """
    _ensure_tool("antechamber")

    if mol.GetNumConformers() == 0:
        raise ValueError("Fragment must have 3D coordinates for charge derivation.")

    if net_charge is None or str(net_charge).strip().lower() == "auto":
        resolved_charge = _formal_charge(mol)
    else:
        resolved_charge = int(net_charge)

    import tempfile
    if work_dir is None:
        tmp = tempfile.mkdtemp(prefix="polycsp_frag_")
        wd = Path(tmp)
    else:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)

    pdb_path = wd / "fragment.pdb"
    mol2_path = wd / "fragment.mol2"
    log_path = wd / "antechamber.log"

    write_pdb_from_rdkit(mol, pdb_path)
    _run_antechamber(
        input_pdb=pdb_path,
        output_mol2=mol2_path,
        charge_model=charge_model,
        net_charge=resolved_charge,
        cwd=wd,
        log_path=log_path,
    )

    if not mol2_path.exists() or mol2_path.stat().st_size == 0:
        raise RuntimeError(
            f"antechamber did not produce {mol2_path}. See: {log_path}"
        )

    charges = _parse_mol2_charges(mol2_path)
    if len(charges) != mol.GetNumAtoms():
        raise RuntimeError(
            f"Charge count mismatch: got {len(charges)} from mol2, "
            f"expected {mol.GetNumAtoms()} atoms."
        )
    return charges


def replicate_charges(
    fragment_charges: List[float],
    n_copies: int,
) -> List[float]:
    """Replicate a fragment charge template n_copies times.

    Every copy gets identical charges, ensuring per-unit symmetry.
    """
    if n_copies < 1:
        raise ValueError(f"n_copies must be >= 1, got {n_copies}")
    return list(fragment_charges) * n_copies


def neutralize_charges(
    charges: List[float],
    target_charge: int = 0,
) -> List[float]:
    """Adjust charges so total equals target_charge (nearest integer).

    Distributes the residual uniformly across all atoms, which is the
    standard RESP/BCC post-processing convention.
    """
    total = sum(charges)
    residual = float(target_charge) - total
    per_atom = residual / float(len(charges))
    return [q + per_atom for q in charges]


def write_charge_template(
    charges: List[float],
    atom_names: List[str],
    path: Path,
    label: str = "fragment",
) -> None:
    """Write a charge template file (JSON) for later reuse."""
    data = {
        "label": label,
        "n_atoms": len(charges),
        "charges": charges,
        "atom_names": atom_names,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_charge_template(path: Path) -> Dict:
    """Read a previously saved charge template."""
    return json.loads(path.read_text(encoding="utf-8"))
