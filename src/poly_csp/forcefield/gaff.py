# poly_csp/forcefield/gaff.py
"""GAFF2 selector fragment parameterization and runtime payload extraction."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

from rdkit import Chem

import openmm as mm
from openmm import app as mmapp
from openmm import unit

from poly_csp.structure.hydrogens import complete_with_hydrogens
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.io.rdkit_io import write_mol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectorAtomParams:
    atom_name: str
    charge_e: float
    sigma_nm: float
    epsilon_kj_per_mol: float


@dataclass(frozen=True)
class SelectorBondTemplate:
    atom_names: tuple[str, str]
    length_nm: float
    k_kj_per_mol_nm2: float


@dataclass(frozen=True)
class SelectorAngleTemplate:
    atom_names: tuple[str, str, str]
    theta0_rad: float
    k_kj_per_mol_rad2: float


@dataclass(frozen=True)
class SelectorTorsionTemplate:
    atom_names: tuple[str, str, str, str]
    periodicity: int
    phase_rad: float
    k_kj_per_mol: float


@dataclass(frozen=True)
class SelectorFragmentParams:
    selector_name: str
    atom_params: dict[str, SelectorAtomParams]
    bonds: tuple[SelectorBondTemplate, ...]
    angles: tuple[SelectorAngleTemplate, ...]
    torsions: tuple[SelectorTorsionTemplate, ...]
    source_prmtop: str | None = None
    fragment_atom_count: int | None = None


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


def _selector_atom_sources(selector_template: SelectorTemplate) -> dict[int, str]:
    connector_heavy = {int(idx) for idx in selector_template.connector_local_roles}
    sources: dict[int, str] = {}
    for atom in selector_template.mol.GetAtoms():
        idx = int(atom.GetIdx())
        if selector_template.attach_dummy_idx is not None and idx == selector_template.attach_dummy_idx:
            sources[idx] = "dummy"
            continue
        if idx in connector_heavy:
            sources[idx] = "connector"
            continue
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 1:
            parent_idx = int(atom.GetNeighbors()[0].GetIdx())
            if parent_idx in connector_heavy:
                sources[idx] = "connector"
                continue
        sources[idx] = "selector"
    return sources


def selector_atom_names(selector_template: SelectorTemplate) -> dict[int, str]:
    hydrogen_by_parent: dict[int, list[int]] = {}
    for atom in selector_template.mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            continue
        parent_idx = int(atom.GetNeighbors()[0].GetIdx())
        hydrogen_by_parent.setdefault(parent_idx, []).append(int(atom.GetIdx()))
    for indices in hydrogen_by_parent.values():
        indices.sort()

    names: dict[int, str] = {}
    for atom in selector_template.mol.GetAtoms():
        idx = int(atom.GetIdx())
        if selector_template.attach_dummy_idx is not None and idx == selector_template.attach_dummy_idx:
            names[idx] = "DUMM"
            continue
        if atom.GetAtomicNum() > 1:
            names[idx] = f"S{idx:03d}"
            continue
        if atom.GetDegree() != 1:
            names[idx] = f"H{idx:03d}"[-4:]
            continue
        parent_idx = int(atom.GetNeighbors()[0].GetIdx())
        siblings = hydrogen_by_parent.get(parent_idx, [idx])
        serial = siblings.index(idx) + 1 if idx in siblings else 1
        names[idx] = f"H{parent_idx % 100:02d}{serial}"
    return names


def _prepare_selector_reference(
    selector_template: SelectorTemplate,
) -> tuple[Chem.Mol, dict[int, str], dict[int, str]]:
    """Build a dummy-free, hydrogen-complete selector reference fragment.

    The selector template carries the future attachment as a dummy atom. For GAFF
    extraction we first convert that into a complete standalone molecule, then
    partition selector vs connector ownership on the finished reference.
    """
    original = Chem.Mol(selector_template.mol)
    original_names = selector_atom_names(selector_template)
    original_sources = _selector_atom_sources(selector_template)

    working = Chem.RWMol(original)
    if selector_template.attach_dummy_idx is not None:
        dummy_idx = int(selector_template.attach_dummy_idx)
        if dummy_idx < 0 or dummy_idx >= working.GetNumAtoms():
            raise ValueError(
                f"Selector dummy index {dummy_idx} is out of range for {selector_template.name!r}."
            )
        dummy_atom = working.GetAtomWithIdx(dummy_idx)
        if dummy_atom.GetDegree() != 1:
            raise ValueError(
                f"Selector dummy atom for {selector_template.name!r} must have degree 1."
            )
        dummy_atom.SetAtomicNum(1)
        dummy_atom.SetFormalCharge(0)
        dummy_atom.SetNoImplicit(True)
        dummy_atom.SetNumExplicitHs(0)
    working = working.GetMol()
    Chem.SanitizeMol(working)
    prepared = complete_with_hydrogens(working, add_coords=True, optimize="h_only")
    names_by_idx: dict[int, str] = {}
    sources_by_idx: dict[int, str] = {}
    for old_idx in range(original.GetNumAtoms()):
        if (
            selector_template.attach_dummy_idx is not None
            and old_idx == int(selector_template.attach_dummy_idx)
        ):
            parent_idx = int(original.GetAtomWithIdx(old_idx).GetNeighbors()[0].GetIdx())
            siblings = [
                int(nbr.GetIdx())
                for nbr in original.GetAtomWithIdx(parent_idx).GetNeighbors()
                if nbr.GetAtomicNum() == 1 and nbr.GetIdx() != old_idx
            ]
            serial = len(siblings) + 1
            names_by_idx[old_idx] = f"H{parent_idx % 100:02d}{serial}"
            sources_by_idx[old_idx] = original_sources[parent_idx]
            continue
        names_by_idx[old_idx] = original_names[old_idx]
        sources_by_idx[old_idx] = original_sources[old_idx]

    hydrogen_by_parent: dict[int, list[int]] = {}
    for atom in prepared.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            continue
        parent_idx = int(atom.GetNeighbors()[0].GetIdx())
        hydrogen_by_parent.setdefault(parent_idx, []).append(int(atom.GetIdx()))
    for indices in hydrogen_by_parent.values():
        indices.sort()

    for atom in prepared.GetAtoms():
        idx = int(atom.GetIdx())
        if idx in names_by_idx:
            continue
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            raise ValueError(
                "Prepared selector reference introduced an unnamed non-hydrogen atom."
            )
        parent_idx = int(atom.GetNeighbors()[0].GetIdx())
        parent_name = names_by_idx.get(parent_idx)
        if parent_name is None or not parent_name.startswith("S"):
            raise ValueError("Prepared selector reference is missing a named heavy parent.")
        local_idx = int(parent_name[1:])
        siblings = hydrogen_by_parent.get(parent_idx, [idx])
        serial = siblings.index(idx) + 1 if idx in siblings else 1
        names_by_idx[idx] = f"H{local_idx % 100:02d}{serial}"
        sources_by_idx[idx] = sources_by_idx[parent_idx]

    if len(names_by_idx) != prepared.GetNumAtoms():
        raise ValueError("Prepared selector reference naming did not cover every atom.")

    return prepared, names_by_idx, sources_by_idx


def parameterize_gaff_fragment(
    fragment_mol: Chem.Mol,
    charge_model: str = "bcc",
    net_charge: int = 0,
    residue_name: str = "FRG",
    input_name: str = "fragment.mol",
    mol2_name: str = "fragment.mol2",
    frcmod_name: str = "fragment.frcmod",
    lib_name: str = "fragment.lib",
    work_dir: Path | None = None,
    ensure_tools_fn: Callable[[Sequence[str]], None] = _ensure_required_tools,
    run_command_fn: Callable[[Sequence[str], Path, Path], None] = _run_command,
    write_input_fn: Callable[[Chem.Mol, str | Path], None] = write_mol,
) -> Dict[str, str]:
    """Run antechamber + parmchk2 + tleap saveoff on a single GAFF fragment.

    The input handoff uses MDL MOL rather than PDB so aromatic bond orders are
    preserved into antechamber atom typing.
    """
    ensure_tools_fn(("antechamber", "parmchk2", "tleap"))

    if work_dir is None:
        wd = Path(tempfile.mkdtemp(prefix="polycsp_gaff_"))
    else:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)

    input_path = wd / input_name
    mol2_path = wd / mol2_name
    frcmod_path = wd / frcmod_name
    lib_path = wd / lib_name

    clean_mol = Chem.RWMol(fragment_mol)
    for atom in clean_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
    clean_mol = clean_mol.GetMol()
    Chem.SanitizeMol(clean_mol)
    prepared = complete_with_hydrogens(clean_mol, add_coords=True, optimize="h_only")
    write_input_fn(prepared, input_path)

    run_command_fn(
        [
            "antechamber",
            "-i", input_path.name, "-fi", "mdl",
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


def _bond_key(atom_names: tuple[str, str]) -> tuple[str, str]:
    a, b = atom_names
    return (a, b) if a <= b else (b, a)


def _angle_key(atom_names: tuple[str, str, str]) -> tuple[str, str, str]:
    a, b, c = atom_names
    return (a, b, c) if a <= c else (c, b, a)


def _append_unique_torsion(
    bucket: list[SelectorTorsionTemplate],
    template: SelectorTorsionTemplate,
) -> None:
    for existing in bucket:
        if existing.atom_names != template.atom_names:
            continue
        if (
            existing.periodicity == template.periodicity
            and abs(existing.phase_rad - template.phase_rad) <= 1e-8
            and abs(existing.k_kj_per_mol - template.k_kj_per_mol) <= 1e-8
        ):
            return
    bucket.append(template)


def parameterize_isolated_selector(
    selector_template: SelectorTemplate,
    charge_model: str = "bcc",
    net_charge: int = 0,
    work_dir: str | Path | None = None,
) -> Dict[str, str]:
    """Parameterize a standalone selector fragment and return GAFF artifacts."""
    prepared, atom_names, _ = _prepare_selector_reference(selector_template)
    artifacts = parameterize_gaff_fragment(
        fragment_mol=prepared,
        charge_model=charge_model,
        net_charge=net_charge,
        residue_name="SEL",
        input_name="selector.mol",
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


def load_selector_fragment_params(
    selector_template: SelectorTemplate,
    charge_model: str = "bcc",
    net_charge: int = 0,
    work_dir: str | Path | None = None,
) -> SelectorFragmentParams:
    """Extract a reusable GAFF2 selector-core payload from the complete fragment."""
    prepared, atom_names, sources = _prepare_selector_reference(selector_template)
    artifacts = parameterize_isolated_selector(
        selector_template=selector_template,
        charge_model=charge_model,
        net_charge=net_charge,
        work_dir=work_dir,
    )

    prmtop = mmapp.AmberPrmtopFile(str(artifacts["prmtop"]))
    system = prmtop.createSystem()

    name_to_idx = {name: idx for idx, name in atom_names.items()}
    prmtop_atoms = list(prmtop.topology.atoms())
    if len(prmtop_atoms) != prepared.GetNumAtoms():
        raise ValueError(
            "Selector GAFF reference atom count changed during AMBER conversion: "
            f"prepared={prepared.GetNumAtoms()}, prmtop={len(prmtop_atoms)}."
        )
    prmtop_idx_to_name = {
        int(atom_idx): atom_names[int(atom_idx)]
        for atom_idx in range(len(prmtop_atoms))
    }

    selector_atom_names_only = {
        atom_names[idx]
        for idx, source in sources.items()
        if source == "selector"
    }

    nonbonded = None
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if isinstance(force, mm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise ValueError("Selector GAFF reference system is missing NonbondedForce.")

    atom_params: dict[str, SelectorAtomParams] = {}
    for atom_idx, atom_name in prmtop_idx_to_name.items():
        local_idx = name_to_idx[atom_name]
        if sources.get(local_idx) != "selector":
            continue
        charge, sigma, epsilon = nonbonded.getParticleParameters(int(atom_idx))
        atom_params[atom_name] = SelectorAtomParams(
            atom_name=atom_name,
            charge_e=float(charge.value_in_unit(unit.elementary_charge)),
            sigma_nm=float(sigma.value_in_unit(unit.nanometer)),
            epsilon_kj_per_mol=float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
        )

    bonds: dict[tuple[str, str], SelectorBondTemplate] = {}
    angles: dict[tuple[str, str, str], SelectorAngleTemplate] = {}
    torsions: list[SelectorTorsionTemplate] = []

    def _all_selector(names: Sequence[str]) -> bool:
        return all(name in selector_atom_names_only for name in names)

    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)

        if isinstance(force, mm.HarmonicBondForce):
            for bond_idx in range(force.getNumBonds()):
                a, b, r0, k = force.getBondParameters(bond_idx)
                names = (
                    prmtop_idx_to_name.get(int(a)),
                    prmtop_idx_to_name.get(int(b)),
                )
                if any(name is None for name in names):
                    continue
                if not _all_selector(names):  # type: ignore[arg-type]
                    continue
                key = _bond_key((str(names[0]), str(names[1])))
                bonds[key] = SelectorBondTemplate(
                    atom_names=key,
                    length_nm=float(r0.value_in_unit(unit.nanometer)),
                    k_kj_per_mol_nm2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                    ),
                )
            continue

        if isinstance(force, mm.HarmonicAngleForce):
            for angle_idx in range(force.getNumAngles()):
                a, b, c, theta0, k = force.getAngleParameters(angle_idx)
                names = (
                    prmtop_idx_to_name.get(int(a)),
                    prmtop_idx_to_name.get(int(b)),
                    prmtop_idx_to_name.get(int(c)),
                )
                if any(name is None for name in names):
                    continue
                if not _all_selector(names):  # type: ignore[arg-type]
                    continue
                key = _angle_key((str(names[0]), str(names[1]), str(names[2])))
                angles[key] = SelectorAngleTemplate(
                    atom_names=key,
                    theta0_rad=float(theta0.value_in_unit(unit.radian)),
                    k_kj_per_mol_rad2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                    ),
                )
            continue

        if isinstance(force, mm.PeriodicTorsionForce):
            for torsion_idx in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, k = force.getTorsionParameters(torsion_idx)
                names = (
                    prmtop_idx_to_name.get(int(a)),
                    prmtop_idx_to_name.get(int(b)),
                    prmtop_idx_to_name.get(int(c)),
                    prmtop_idx_to_name.get(int(d)),
                )
                if any(name is None for name in names):
                    continue
                if not _all_selector(names):  # type: ignore[arg-type]
                    continue
                _append_unique_torsion(
                    torsions,
                    SelectorTorsionTemplate(
                        atom_names=(str(names[0]), str(names[1]), str(names[2]), str(names[3])),
                        periodicity=int(periodicity),
                        phase_rad=float(phase.value_in_unit(unit.radian)),
                        k_kj_per_mol=float(k.value_in_unit(unit.kilojoule_per_mole)),
                    ),
                )

    if set(atom_params) != selector_atom_names_only:
        missing = sorted(selector_atom_names_only.difference(atom_params))
        extra = sorted(set(atom_params).difference(selector_atom_names_only))
        raise ValueError(
            "Selector GAFF payload atom coverage mismatch. "
            f"Missing={missing}, extra={extra}."
        )

    return SelectorFragmentParams(
        selector_name=selector_template.name,
        atom_params=atom_params,
        bonds=tuple(sorted(bonds.values(), key=lambda item: item.atom_names)),
        angles=tuple(sorted(angles.values(), key=lambda item: item.atom_names)),
        torsions=tuple(
            sorted(
                torsions,
                key=lambda item: (
                    item.atom_names,
                    item.periodicity,
                    item.phase_rad,
                    item.k_kj_per_mol,
                ),
            )
        ),
        source_prmtop=str(artifacts["prmtop"]),
        fragment_atom_count=prepared.GetNumAtoms(),
    )
