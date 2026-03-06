# poly_csp/forcefield/gaff.py
"""Transfer GAFF2 bonded forces from a selector prmtop to the full molecule.

This module solves the per-fragment mapping problem: each selector instance
in the polymer is an identical copy of the template, so we load the prmtop
once and replicate its forces for every instance, remapping atom indices
through the ``_poly_csp_selector_local_idx`` property.

Dummy atom handling
-------------------
The selector template SMILES includes a ``[*]`` dummy atom at
``attach_dummy_idx``.  During GAFF2 parameterisation the dummy is replaced
with H, so the prmtop has the *same atom count* as the original template.
After attachment to the polymer the dummy atom is **removed**, so the
corresponding ``local_idx`` is absent from the per-instance mapping.
Any prmtop force term that involves the dummy index is skipped.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Sequence

from rdkit import Chem

import openmm as mm
from openmm import app as mmapp

from poly_csp.topology.atom_mapping import selector_instance_maps
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.io.pdb import write_pdb_from_rdkit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AmberTools fragment parameterization helpers
# ---------------------------------------------------------------------------


def _ensure_required_tools(tools: Sequence[str]) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            "GAFF fragment parameterization requires executables not found on PATH: "
            + ", ".join(missing)
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


def parameterize_gaff_fragment(
    fragment_mol: Chem.Mol,
    charge_model: str = "bcc",
    net_charge: int = 0,
    residue_name: str = "FRG",
    pdb_name: str = "fragment.pdb",
    mol2_name: str = "fragment.mol2",
    frcmod_name: str = "fragment.frcmod",
    lib_name: str = "fragment.lib",
    work_dir: Path | None = None,
    ensure_tools_fn: Callable[[Sequence[str]], None] = _ensure_required_tools,
    run_command_fn: Callable[[Sequence[str], Path, Path], None] = _run_command,
    write_pdb_fn: Callable[[Chem.Mol, str | Path], None] = write_pdb_from_rdkit,
) -> Dict[str, str]:
    """Run antechamber + parmchk2 + tleap saveoff on a single GAFF fragment."""
    ensure_tools_fn(("antechamber", "parmchk2"))

    if work_dir is None:
        wd = Path(tempfile.mkdtemp(prefix="polycsp_gaff_"))
    else:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)

    pdb_path = wd / pdb_name
    mol2_path = wd / mol2_name
    frcmod_path = wd / frcmod_name
    lib_path = wd / lib_name

    # Replace dummy atoms ([*], atomic number 0) with hydrogen so that
    # antechamber can perform atom typing and charge derivation.
    clean_mol = Chem.RWMol(fragment_mol)
    for atom in clean_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
    Chem.SanitizeMol(clean_mol)

    write_pdb_fn(clean_mol, pdb_path)

    run_command_fn(
        [
            "antechamber",
            "-i", pdb_path.name, "-fi", "pdb",
            "-o", mol2_path.name, "-fo", "mol2",
            "-at", "gaff2", "-c", charge_model,
            "-nc", str(net_charge),
            "-rn", residue_name, "-dr", "no", "-pf", "y", "-s", "2",
        ],
        cwd=wd,
        log_path=wd / "antechamber.log",
    )
    run_command_fn(
        [
            "parmchk2",
            "-i", mol2_path.name, "-f", "mol2",
            "-s", "gaff2", "-o", frcmod_path.name,
        ],
        cwd=wd,
        log_path=wd / "parmchk2.log",
    )

    saveoff_script = "\n".join([
        "source leaprc.gaff2",
        f"loadamberparams {frcmod_path.name}",
        f"frag = loadmol2 {mol2_path.name}",
        f"saveoff frag {lib_path.name}",
        "quit",
    ]) + "\n"
    saveoff_in = wd / "saveoff.in"
    saveoff_in.write_text(saveoff_script, encoding="utf-8")
    run_command_fn(
        ["tleap", "-f", saveoff_in.name],
        cwd=wd,
        log_path=wd / "saveoff.log",
    )

    return {
        "mol2": str(mol2_path.resolve()),
        "frcmod": str(frcmod_path.resolve()),
        "lib": str(lib_path.resolve()),
    }


def build_fragment_prmtop(
    mol2_path: str | Path,
    frcmod_path: str | Path,
    prmtop_name: str = "fragment.prmtop",
    inpcrd_name: str = "fragment.inpcrd",
    clean_mol2_name: str = "fragment_clean.mol2",
    work_dir: Path | None = None,
    run_command_fn: Callable[[Sequence[str], Path, Path], None] = _run_command,
) -> str:
    """Create a standalone AMBER prmtop for a GAFF fragment."""
    mol2_path = Path(mol2_path)
    frcmod_path = Path(frcmod_path)

    if work_dir is None:
        work_dir = mol2_path.parent
    else:
        work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    clean_mol2 = work_dir / clean_mol2_name
    _deduplicate_mol2_bonds(mol2_path, clean_mol2)

    prmtop_path = work_dir / prmtop_name
    inpcrd_path = work_dir / inpcrd_name
    script = "\n".join([
        "source leaprc.gaff2",
        f"loadamberparams {frcmod_path.resolve()}",
        f"frag = loadmol2 {clean_mol2.resolve()}",
        f"saveamberparm frag {prmtop_path.name} {inpcrd_path.name}",
        "quit",
    ]) + "\n"
    script_path = work_dir / "build_prmtop.in"
    script_path.write_text(script, encoding="utf-8")

    run_command_fn(
        ["tleap", "-f", script_path.name],
        cwd=work_dir,
        log_path=work_dir / "build_prmtop.log",
    )
    if not prmtop_path.exists() or prmtop_path.stat().st_size == 0:
        raise RuntimeError(
            f"tleap failed to generate fragment prmtop. "
            f"See: {work_dir / 'build_prmtop.log'}"
        )
    return str(prmtop_path.resolve())


def _deduplicate_mol2_bonds(src: Path, dst: Path) -> None:
    """Remove duplicate bonds from an AMBER mol2 file."""
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    out_lines: list[str] = []
    in_bond_section = False
    past_bond_section = False
    bond_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "@<TRIPOS>BOND":
            in_bond_section = True
            out_lines.append(line)
            continue
        if in_bond_section and stripped.startswith("@<TRIPOS>"):
            in_bond_section = False
            past_bond_section = True
            _flush_dedup_bonds(bond_lines, out_lines)
            out_lines.append(line)
            continue
        if in_bond_section:
            bond_lines.append(line)
        else:
            out_lines.append(line)

    if in_bond_section and not past_bond_section:
        _flush_dedup_bonds(bond_lines, out_lines)

    _fix_mol2_bond_count(out_lines, bond_lines)
    dst.write_text("".join(out_lines), encoding="utf-8")


def _flush_dedup_bonds(bond_lines: list[str], out_lines: list[str]) -> None:
    """Deduplicate bond lines and append to out_lines with renumbered IDs."""
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int, str]] = []
    for raw_line in bond_lines:
        parts = raw_line.split()
        if len(parts) < 4:
            continue
        a, b = int(parts[1]), int(parts[2])
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((a, b, parts[3]))

    for idx, (a, b, btype) in enumerate(deduped, 1):
        out_lines.append(f"     {idx:>2d}    {a:>2d}    {b:>2d} {btype}\n")

    bond_lines.clear()
    bond_lines.extend([f"{len(deduped)}"])


def _fix_mol2_bond_count(lines: list[str], count_stash: list[str]) -> None:
    """Fix the bond count in the @<TRIPOS>MOLECULE header."""
    if not count_stash:
        return
    try:
        new_count = int(count_stash[0])
    except (ValueError, IndexError):
        return

    for i, line in enumerate(lines):
        if line.strip() == "@<TRIPOS>MOLECULE":
            if i + 2 < len(lines):
                parts = lines[i + 2].split()
                if len(parts) >= 2:
                    parts[1] = str(new_count)
                    lines[i + 2] = "   " + "    ".join(parts) + "\n"
            break


# ---------------------------------------------------------------------------
# Load GAFF2 forces from prmtop and remap to molecule indices
# ---------------------------------------------------------------------------

def load_gaff2_selector_forces(
    selector_prmtop_path: str,
    mol: Chem.Mol,
    selector_template: SelectorTemplate,
) -> list[mm.Force]:
    """Load GAFF2 bonded forces from the selector prmtop and remap indices.

    Returns a list of OpenMM Force objects (HarmonicBondForce,
    HarmonicAngleForce, PeriodicTorsionForce) whose particle indices
    reference the full polymer molecule.

    Parameters
    ----------
    selector_prmtop_path
        Path to an AMBER prmtop file for the standalone selector fragment
        (with dummy atom replaced by H).
    mol
        The full polymer RDKit molecule with selector atoms annotated via
        ``_poly_csp_selector_instance`` and ``_poly_csp_selector_local_idx``.
    selector_template
        The SelectorTemplate that defines ``attach_dummy_idx``.

    Returns
    -------
    List of OpenMM Force objects ready to be added to a System.  Typically
    contains one HarmonicBondForce, one HarmonicAngleForce, and one or two
    PeriodicTorsionForce objects (proper torsions + impropers). Terms that
    cross into connector atoms are intentionally absent because
    ``selector_instance_maps`` includes selector-core atoms only; those
    linkage terms are handled by connector overlays.
    """
    dummy_idx = selector_template.attach_dummy_idx  # may be None

    # 1. Load the prmtop and create an OpenMM system to extract the forces.
    prmtop = mmapp.AmberPrmtopFile(selector_prmtop_path)
    ref_system = prmtop.createSystem()

    # 2. Extract per-instance mappings: {instance_id → {local_idx → global_idx}}.
    #    local_idx == prmtop atom index (except for the dummy, which is absent).
    mappings = selector_instance_maps(mol)
    if not mappings:
        log.warning("No selector instances found in molecule; returning empty forces.")
        return []

    n_instances = len(mappings)
    log.info(
        "Transferring GAFF2 forces from %s to %d selector instance(s).",
        selector_prmtop_path, n_instances,
    )

    # 3. For each force type in the reference system, create a corresponding
    #    force for the full molecule and replicate every term for each instance.
    out_forces: list[mm.Force] = []

    for force_idx in range(ref_system.getNumForces()):
        ref_force = ref_system.getForce(force_idx)

        if isinstance(ref_force, mm.HarmonicBondForce):
            out_bond = mm.HarmonicBondForce()
            for inst_map in mappings.values():
                _transfer_bonds(ref_force, inst_map, dummy_idx, out_bond)
            out_forces.append(out_bond)
            log.info("  HarmonicBondForce: %d bonds", out_bond.getNumBonds())

        elif isinstance(ref_force, mm.HarmonicAngleForce):
            out_angle = mm.HarmonicAngleForce()
            for inst_map in mappings.values():
                _transfer_angles(ref_force, inst_map, dummy_idx, out_angle)
            out_forces.append(out_angle)
            log.info("  HarmonicAngleForce: %d angles", out_angle.getNumAngles())

        elif isinstance(ref_force, mm.PeriodicTorsionForce):
            out_torsion = mm.PeriodicTorsionForce()
            for inst_map in mappings.values():
                _transfer_torsions(ref_force, inst_map, dummy_idx, out_torsion)
            out_forces.append(out_torsion)
            log.info("  PeriodicTorsionForce: %d torsions", out_torsion.getNumTorsions())

        # NonbondedForce, CMMotionRemover, etc. are intentionally skipped —
        # we use soft repulsion instead.

    return out_forces


def _remap_idx(prmtop_idx: int, inst_map: Dict[int, int], dummy_idx: int | None) -> int | None:
    """Map a prmtop atom index to a global molecule index.

    Returns None if the atom is the dummy (not present in the polymer).
    The mapping is straightforward: prmtop indices correspond 1:1 to
    template local_idx values, and the instance mapping gives us global_idx.
    """
    if dummy_idx is not None and prmtop_idx == dummy_idx:
        return None
    local_idx = prmtop_idx
    return inst_map.get(local_idx)


def _transfer_bonds(
    ref_force: mm.HarmonicBondForce,
    inst_map: Dict[int, int],
    dummy_idx: int | None,
    out_force: mm.HarmonicBondForce,
) -> None:
    """Copy bond terms from the reference force, remapping indices."""
    for bi in range(ref_force.getNumBonds()):
        p1, p2, r0, k = ref_force.getBondParameters(bi)
        g1 = _remap_idx(p1, inst_map, dummy_idx)
        g2 = _remap_idx(p2, inst_map, dummy_idx)
        if g1 is None or g2 is None:
            continue  # involves dummy atom → skip
        out_force.addBond(g1, g2, r0, k)


def _transfer_angles(
    ref_force: mm.HarmonicAngleForce,
    inst_map: Dict[int, int],
    dummy_idx: int | None,
    out_force: mm.HarmonicAngleForce,
) -> None:
    """Copy angle terms from the reference force, remapping indices."""
    for ai in range(ref_force.getNumAngles()):
        p1, p2, p3, theta0, k = ref_force.getAngleParameters(ai)
        g1 = _remap_idx(p1, inst_map, dummy_idx)
        g2 = _remap_idx(p2, inst_map, dummy_idx)
        g3 = _remap_idx(p3, inst_map, dummy_idx)
        if g1 is None or g2 is None or g3 is None:
            continue
        out_force.addAngle(g1, g2, g3, theta0, k)


def _transfer_torsions(
    ref_force: mm.PeriodicTorsionForce,
    inst_map: Dict[int, int],
    dummy_idx: int | None,
    out_force: mm.PeriodicTorsionForce,
) -> None:
    """Copy torsion terms from the reference force, remapping indices."""
    for ti in range(ref_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = ref_force.getTorsionParameters(ti)
        g1 = _remap_idx(p1, inst_map, dummy_idx)
        g2 = _remap_idx(p2, inst_map, dummy_idx)
        g3 = _remap_idx(p3, inst_map, dummy_idx)
        g4 = _remap_idx(p4, inst_map, dummy_idx)
        if g1 is None or g2 is None or g3 is None or g4 is None:
            continue
        out_force.addTorsion(g1, g2, g3, g4, periodicity, phase, k)


def parameterize_isolated_selector(
    selector_mol: Chem.Mol,
    charge_model: str = "bcc",
    net_charge: int = 0,
    work_dir: str | Path | None = None,
) -> Dict[str, str]:
    """Parameterize a standalone selector fragment and return GAFF artifacts."""
    artifacts = parameterize_gaff_fragment(
        fragment_mol=selector_mol,
        charge_model=charge_model,
        net_charge=net_charge,
        residue_name="SEL",
        pdb_name="selector.pdb",
        mol2_name="selector.mol2",
        frcmod_name="selector.frcmod",
        lib_name="selector.lib",
        work_dir=None if work_dir is None else Path(work_dir),
    )
    prmtop = build_fragment_prmtop(
        mol2_path=artifacts["mol2"],
        frcmod_path=artifacts["frcmod"],
        prmtop_name="selector.prmtop",
        inpcrd_name="selector.inpcrd",
        clean_mol2_name="selector_clean.mol2",
        work_dir=Path(work_dir) if work_dir is not None else None,
    )
    out = dict(artifacts)
    out["prmtop"] = prmtop
    return out
