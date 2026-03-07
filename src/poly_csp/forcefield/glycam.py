"""Runtime GLYCAM reference extraction for pure polysaccharide backbones."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, List, Literal, Sequence

from poly_csp.config.schema import EndMode, MonomerRepresentation, PolymerKind
from poly_csp.forcefield.payload_cache import (
    glycam_cache_dir,
    glycam_cache_identity,
    load_cached_glycam_params,
    store_cached_glycam_params,
)


ResidueRole = Literal["terminal_reducing", "internal", "terminal_nonreducing"]


# GLYCAM06j residue codes for poly_csp residue order.
# poly_csp indexes residues from the free-C1 end toward the free-O4 end, so
# open-chain amylose/cellulose sequences are 4G* ... 4G* 0G*.
GLYCAM_RESIDUE_NAMES = {
    ("amylose", "terminal_reducing"): "4GA",
    ("amylose", "internal"): "4GA",
    ("amylose", "terminal_nonreducing"): "0GA",
    ("cellulose", "terminal_reducing"): "4GB",
    ("cellulose", "internal"): "4GB",
    ("cellulose", "terminal_nonreducing"): "0GB",
}


@dataclass(frozen=True, order=True)
class GlycamAtomToken:
    residue_offset: int
    atom_name: str


@dataclass(frozen=True)
class GlycamAtomParams:
    charge_e: float
    sigma_nm: float
    epsilon_kj_per_mol: float
    residue_name: str
    source_atom_name: str


@dataclass(frozen=True)
class GlycamBondTemplate:
    atoms: tuple[GlycamAtomToken, GlycamAtomToken]
    length_nm: float
    k_kj_per_mol_nm2: float


@dataclass(frozen=True)
class GlycamAngleTemplate:
    atoms: tuple[GlycamAtomToken, GlycamAtomToken, GlycamAtomToken]
    theta0_rad: float
    k_kj_per_mol_rad2: float


@dataclass(frozen=True)
class GlycamTorsionTemplate:
    atoms: tuple[GlycamAtomToken, GlycamAtomToken, GlycamAtomToken, GlycamAtomToken]
    periodicity: int
    phase_rad: float
    k_kj_per_mol: float


@dataclass(frozen=True)
class GlycamResidueTemplate:
    residue_role: ResidueRole
    residue_name: str
    atom_names: tuple[str, ...]
    bonds: tuple[GlycamBondTemplate, ...]
    angles: tuple[GlycamAngleTemplate, ...]
    torsions: tuple[GlycamTorsionTemplate, ...]


@dataclass(frozen=True)
class GlycamLinkageTemplate:
    residue_roles: tuple[ResidueRole, ResidueRole]
    bonds: tuple[GlycamBondTemplate, ...]
    angles: tuple[GlycamAngleTemplate, ...]
    torsions: tuple[GlycamTorsionTemplate, ...]


@dataclass(frozen=True)
class GlycamParams:
    polymer: PolymerKind
    representation: MonomerRepresentation
    end_mode: EndMode
    atom_params: dict[tuple[ResidueRole, str], GlycamAtomParams]
    residue_templates: dict[ResidueRole, GlycamResidueTemplate]
    linkage_templates: dict[tuple[ResidueRole, ResidueRole], GlycamLinkageTemplate]
    supported_states: tuple[tuple[str, str, str, str], ...]
    provenance: dict[str, object]


def glycam_residue_roles_for_dp(dp: int) -> list[ResidueRole]:
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}")
    if dp == 1:
        return ["terminal_nonreducing"]
    return ["terminal_reducing", *["internal"] * (dp - 2), "terminal_nonreducing"]


def build_glycam_sequence(polymer: str, dp: int) -> List[str]:
    """Build the GLYCAM residue-code sequence for the poly_csp residue order."""
    key_base = polymer.lower()
    roles = glycam_residue_roles_for_dp(dp)
    try:
        return [GLYCAM_RESIDUE_NAMES[(key_base, role)] for role in roles]
    except KeyError as exc:
        raise ValueError(f"Unsupported GLYCAM polymer {polymer!r}.") from exc


def _ensure_required_tools(tools: Sequence[str]) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "GLYCAM reference extraction requires executables not found on PATH: "
            + ", ".join(missing)
            + ". Install AmberTools with GLYCAM06 support."
        )


def _run_command(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
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


def build_tleap_script(
    polymer: str,
    dp: int,
    linkage_frcmod_path: str | None = None,
    model_name: str = "model",
    prmtop_name: str | None = None,
    inpcrd_name: str | None = None,
    periodic: bool = False,
    box_vectors_A: tuple[float, float, float] | None = None,
) -> str:
    """Generate a tleap script for pure-backbone GLYCAM reference assembly."""
    prmtop_name = prmtop_name or f"{model_name}.prmtop"
    inpcrd_name = inpcrd_name or f"{model_name}.inpcrd"

    lines: List[str] = [
        "# poly_csp GLYCAM06j backbone reference assembly",
        "source leaprc.GLYCAM_06j-1",
    ]
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


def run_tleap_assembly(
    tleap_script: str,
    outdir: Path,
    model_name: str = "model",
) -> Dict[str, object]:
    """Execute tleap with the given script and return artifact paths."""
    _ensure_required_tools(("tleap",))

    outdir.mkdir(parents=True, exist_ok=True)
    tleap_path = outdir / "tleap.in"
    tleap_log = outdir / "tleap.log"
    prmtop_path = outdir / f"{model_name}.prmtop"
    inpcrd_path = outdir / f"{model_name}.inpcrd"

    tleap_path.write_text(tleap_script, encoding="utf-8")
    _run_command(["tleap", "-f", tleap_path.name], cwd=outdir, log_path=tleap_log)

    missing = [
        str(path)
        for path in (prmtop_path, inpcrd_path)
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "tleap assembly completed but expected outputs were not generated: "
            + ", ".join(missing)
            + f". See: {tleap_log}"
        )

    return {
        "prmtop": str(prmtop_path),
        "inpcrd": str(inpcrd_path),
        "tleap_input": str(tleap_path),
        "tleap_log": str(tleap_log),
        "parameterized": True,
        "parameter_backend": "glycam_reference_extract",
    }


def build_linkage_frcmod(outdir: Path, filename: str = "linkage.frcmod") -> Path:
    """Generate a supplementary frcmod for GLYCAM glycosidic linkage assembly."""
    missing_torsions = [
        "H2-Cg-Cg-H2",
        "H2-Cg-Cg-H1",
        "H1-Cg-Cg-H1",
        "H2-Cg-Cg-Oh",
        "H1-Cg-Cg-Oh",
        "Oh-Cg-Cg-Oh",
        "H2-Cg-Cg-Os",
        "H1-Cg-Cg-Os",
        "Oh-Cg-Cg-Os",
        "Os-Cg-Cg-Os",
        "Cg-Cg-Cg-H2",
        "Cg-Cg-Cg-H1",
        "Cg-Cg-Cg-Oh",
        "Cg-Cg-Cg-Os",
        "Cg-Cg-Cg-Cg",
    ]
    lines = [
        "Supplementary frcmod for glycosidic linkage assembly (poly_csp)",
        "MASS",
        "",
        "BOND",
        "",
        "ANGLE",
        "",
        "DIHE",
    ]
    for torsion in missing_torsions:
        lines.append(f"{torsion}   1    0.000         0.0             3.0")
    lines.extend(["", "IMPROPER", "", "NONBON", "", ""])

    outdir.mkdir(parents=True, exist_ok=True)
    frcmod_path = outdir / filename
    frcmod_path.write_text("\n".join(lines), encoding="utf-8")
    return frcmod_path


def _float_close(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def _bond_key(
    atoms: tuple[GlycamAtomToken, GlycamAtomToken],
) -> tuple[GlycamAtomToken, GlycamAtomToken]:
    a, b = atoms
    return (a, b) if a <= b else (b, a)


def _angle_key(
    atoms: tuple[GlycamAtomToken, GlycamAtomToken, GlycamAtomToken],
) -> tuple[GlycamAtomToken, GlycamAtomToken, GlycamAtomToken]:
    a, b, c = atoms
    return (a, b, c) if a <= c else (c, b, a)


def _store_atom_params(
    bucket: dict[tuple[ResidueRole, str], GlycamAtomParams],
    role: ResidueRole,
    atom_name: str,
    params: GlycamAtomParams,
) -> None:
    key = (role, atom_name)
    existing = bucket.get(key)
    if existing is None:
        bucket[key] = params
        return
    if not (
        _float_close(existing.charge_e, params.charge_e)
        and _float_close(existing.sigma_nm, params.sigma_nm)
        and _float_close(existing.epsilon_kj_per_mol, params.epsilon_kj_per_mol)
        and existing.residue_name == params.residue_name
    ):
        raise ValueError(f"Conflicting GLYCAM atom parameters for key {key!r}.")


def _store_bond(
    bucket: dict[tuple[GlycamAtomToken, GlycamAtomToken], GlycamBondTemplate],
    template: GlycamBondTemplate,
) -> None:
    key = _bond_key(template.atoms)
    existing = bucket.get(key)
    if existing is None:
        bucket[key] = GlycamBondTemplate(
            atoms=key,
            length_nm=template.length_nm,
            k_kj_per_mol_nm2=template.k_kj_per_mol_nm2,
        )
        return
    if not (
        _float_close(existing.length_nm, template.length_nm)
        and _float_close(existing.k_kj_per_mol_nm2, template.k_kj_per_mol_nm2)
    ):
        raise ValueError(f"Conflicting GLYCAM bond template for key {key!r}.")


def _store_angle(
    bucket: dict[
        tuple[GlycamAtomToken, GlycamAtomToken, GlycamAtomToken],
        GlycamAngleTemplate,
    ],
    template: GlycamAngleTemplate,
) -> None:
    key = _angle_key(template.atoms)
    existing = bucket.get(key)
    if existing is None:
        bucket[key] = GlycamAngleTemplate(
            atoms=key,
            theta0_rad=template.theta0_rad,
            k_kj_per_mol_rad2=template.k_kj_per_mol_rad2,
        )
        return
    if not (
        _float_close(existing.theta0_rad, template.theta0_rad)
        and _float_close(existing.k_kj_per_mol_rad2, template.k_kj_per_mol_rad2)
    ):
        raise ValueError(f"Conflicting GLYCAM angle template for key {key!r}.")


def _append_unique_torsion(
    bucket: list[GlycamTorsionTemplate],
    torsion: GlycamTorsionTemplate,
) -> None:
    for existing in bucket:
        if existing.atoms != torsion.atoms:
            continue
        if (
            existing.periodicity == torsion.periodicity
            and _float_close(existing.phase_rad, torsion.phase_rad)
            and _float_close(existing.k_kj_per_mol, torsion.k_kj_per_mol)
        ):
            return
    bucket.append(torsion)


def _tokenized_atoms(
    atom_names: list[str],
    residue_indices: list[int],
) -> tuple[GlycamAtomToken, ...]:
    anchor = min(residue_indices)
    return tuple(
        GlycamAtomToken(
            residue_offset=int(residue_idx - anchor),
            atom_name=str(atom_name),
        )
        for atom_name, residue_idx in zip(atom_names, residue_indices, strict=True)
    )


def _new_residue_bucket(residue_name: str) -> dict[str, object]:
    return {
        "residue_name": residue_name,
        "atom_names": set(),
        "bonds": {},
        "angles": {},
        "torsions": [],
    }


def _new_linkage_bucket() -> dict[str, object]:
    return {"bonds": {}, "angles": {}, "torsions": []}


def _partition_reference_terms(
    prmtop_path: str | Path,
    dp: int,
    atom_params: dict[tuple[ResidueRole, str], GlycamAtomParams],
    residue_terms: dict[ResidueRole, dict[str, object]],
    linkage_terms: dict[tuple[ResidueRole, ResidueRole], dict[str, object]],
) -> None:
    import openmm as mm
    from openmm import app as mmapp
    from openmm import unit

    prmtop = mmapp.AmberPrmtopFile(str(prmtop_path))
    ref_system = prmtop.createSystem()
    roles = glycam_residue_roles_for_dp(dp)
    residues = list(prmtop.topology.residues())
    if len(residues) != len(roles):
        raise ValueError(
            "Reference GLYCAM topology residue count does not match expected roles."
        )

    atom_meta: dict[int, tuple[int, ResidueRole, str, str]] = {}
    for residue, role in zip(residues, roles, strict=True):
        residue_name = str(residue.name).strip()
        bucket = residue_terms.setdefault(role, _new_residue_bucket(residue_name))
        if bucket["residue_name"] != residue_name:
            raise ValueError(f"Conflicting GLYCAM residue name for role {role!r}.")
        for atom in residue.atoms():
            atom_name = str(atom.name).strip()
            atom_meta[int(atom.index)] = (int(residue.index), role, residue_name, atom_name)
            bucket["atom_names"].add(atom_name)

    nonbonded = None
    for force_idx in range(ref_system.getNumForces()):
        force = ref_system.getForce(force_idx)
        if isinstance(force, mm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise ValueError("GLYCAM reference system is missing NonbondedForce.")

    for atom_idx, (_, role, residue_name, atom_name) in atom_meta.items():
        charge, sigma, epsilon = nonbonded.getParticleParameters(int(atom_idx))
        _store_atom_params(
            atom_params,
            role,
            atom_name,
            GlycamAtomParams(
                charge_e=float(charge.value_in_unit(unit.elementary_charge)),
                sigma_nm=float(sigma.value_in_unit(unit.nanometer)),
                epsilon_kj_per_mol=float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
                residue_name=residue_name,
                source_atom_name=atom_name,
            ),
        )

    def _classify(atom_indices: Sequence[int]) -> tuple[str, object, tuple[GlycamAtomToken, ...]]:
        residue_indices = [atom_meta[int(atom_idx)][0] for atom_idx in atom_indices]
        unique_residues = sorted(set(residue_indices))
        atom_names = [atom_meta[int(atom_idx)][3] for atom_idx in atom_indices]

        if len(unique_residues) == 1:
            residue_index = unique_residues[0]
            role = roles[residue_index]
            tokens = _tokenized_atoms(atom_names, [residue_index] * len(atom_indices))
            return "residue", role, tokens

        if len(unique_residues) == 2 and unique_residues[1] == unique_residues[0] + 1:
            left_idx, right_idx = unique_residues
            tokens = _tokenized_atoms(atom_names, residue_indices)
            return "linkage", (roles[left_idx], roles[right_idx]), tokens

        raise ValueError(
            "GLYCAM reference term spans unsupported residue topology: "
            f"{unique_residues!r}"
        )

    for force_idx in range(ref_system.getNumForces()):
        force = ref_system.getForce(force_idx)

        if isinstance(force, mm.HarmonicBondForce):
            for bond_idx in range(force.getNumBonds()):
                a, b, r0, k = force.getBondParameters(bond_idx)
                kind, key, tokens = _classify((int(a), int(b)))
                template = GlycamBondTemplate(
                    atoms=(tokens[0], tokens[1]),
                    length_nm=float(r0.value_in_unit(unit.nanometer)),
                    k_kj_per_mol_nm2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                    ),
                )
                if kind == "residue":
                    _store_bond(residue_terms[key]["bonds"], template)  # type: ignore[index]
                else:
                    bucket = linkage_terms.setdefault(
                        key,  # type: ignore[arg-type]
                        _new_linkage_bucket(),
                    )
                    _store_bond(bucket["bonds"], template)
            continue

        if isinstance(force, mm.HarmonicAngleForce):
            for angle_idx in range(force.getNumAngles()):
                a, b, c, theta0, k = force.getAngleParameters(angle_idx)
                kind, key, tokens = _classify((int(a), int(b), int(c)))
                template = GlycamAngleTemplate(
                    atoms=(tokens[0], tokens[1], tokens[2]),
                    theta0_rad=float(theta0.value_in_unit(unit.radian)),
                    k_kj_per_mol_rad2=float(
                        k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                    ),
                )
                if kind == "residue":
                    _store_angle(residue_terms[key]["angles"], template)  # type: ignore[index]
                else:
                    bucket = linkage_terms.setdefault(
                        key,  # type: ignore[arg-type]
                        _new_linkage_bucket(),
                    )
                    _store_angle(bucket["angles"], template)
            continue

        if isinstance(force, mm.PeriodicTorsionForce):
            for torsion_idx in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, k = force.getTorsionParameters(torsion_idx)
                kind, key, tokens = _classify((int(a), int(b), int(c), int(d)))
                template = GlycamTorsionTemplate(
                    atoms=(tokens[0], tokens[1], tokens[2], tokens[3]),
                    periodicity=int(periodicity),
                    phase_rad=float(phase.value_in_unit(unit.radian)),
                    k_kj_per_mol=float(k.value_in_unit(unit.kilojoule_per_mole)),
                )
                if kind == "residue":
                    _append_unique_torsion(residue_terms[key]["torsions"], template)  # type: ignore[index]
                else:
                    bucket = linkage_terms.setdefault(
                        key,  # type: ignore[arg-type]
                        _new_linkage_bucket(),
                    )
                    _append_unique_torsion(bucket["torsions"], template)


def _build_reference_prmtop(
    polymer: PolymerKind,
    dp: int,
    work_dir: Path | None = None,
) -> dict[str, object]:
    outdir = (
        Path(tempfile.mkdtemp(prefix=f"polycsp_glycam_{polymer}_dp{dp}_"))
        if work_dir is None
        else Path(work_dir)
    )
    outdir.mkdir(parents=True, exist_ok=True)

    linkage_frcmod = build_linkage_frcmod(outdir, filename=f"linkage_dp{dp}.frcmod")
    model_name = f"glycam_ref_dp{dp}"
    script = build_tleap_script(
        polymer=polymer,
        dp=dp,
        linkage_frcmod_path=str(linkage_frcmod.resolve()),
        model_name=model_name,
        prmtop_name=f"{model_name}.prmtop",
        inpcrd_name=f"{model_name}.inpcrd",
    )
    result = run_tleap_assembly(script, outdir=outdir, model_name=model_name)
    result["reference_dp"] = dp
    result["sequence"] = build_glycam_sequence(polymer=polymer, dp=dp)
    return result


def _sorted_torsions(
    torsions: list[GlycamTorsionTemplate],
) -> tuple[GlycamTorsionTemplate, ...]:
    return tuple(
        sorted(
            torsions,
            key=lambda item: (
                item.atoms,
                item.periodicity,
                item.phase_rad,
                item.k_kj_per_mol,
            ),
        )
    )


def _validate_extracted_templates(
    residue_templates: dict[ResidueRole, GlycamResidueTemplate],
    linkage_templates: dict[tuple[ResidueRole, ResidueRole], GlycamLinkageTemplate],
) -> None:
    required_roles = {
        "terminal_reducing",
        "internal",
        "terminal_nonreducing",
    }
    missing_roles = sorted(required_roles.difference(residue_templates))
    if missing_roles:
        raise ValueError(
            "Missing extracted GLYCAM residue templates for roles: "
            + ", ".join(missing_roles)
        )

    required_pairs = {
        ("terminal_reducing", "terminal_nonreducing"),
        ("terminal_reducing", "internal"),
        ("internal", "internal"),
        ("internal", "terminal_nonreducing"),
    }
    missing_pairs = sorted(required_pairs.difference(linkage_templates))
    if missing_pairs:
        raise ValueError(
            "Missing extracted GLYCAM linkage templates for residue-role pairs: "
            + ", ".join(f"{left}->{right}" for left, right in missing_pairs)
        )


_GLYCAM_PARAMS_CACHE: dict[tuple[str, str, str, str | None], GlycamParams] = {}


def _with_cache_provenance(
    params: GlycamParams,
    *,
    cache_enabled: bool,
    cache_entry_dir: Path | None,
    cache_hit: bool,
    cache_kind: str,
) -> GlycamParams:
    provenance = dict(params.provenance)
    provenance["cache"] = {
        "enabled": bool(cache_enabled),
        "entry_dir": (
            str(cache_entry_dir.resolve()) if cache_entry_dir is not None else None
        ),
        "hit": bool(cache_hit),
        "kind": str(cache_kind),
    }
    return replace(params, provenance=provenance)


def load_glycam_params(
    polymer: PolymerKind,
    representation: MonomerRepresentation = "anhydro",
    end_mode: EndMode = "open",
    work_dir: Path | None = None,
    *,
    cache_enabled: bool = False,
    cache_dir: str | Path | None = None,
) -> GlycamParams:
    """Extract reusable GLYCAM templates for the supported pure-backbone slice."""
    if representation != "anhydro":
        raise ValueError(
            "Phase 2 GLYCAM loading currently supports only the anhydro backbone representation."
        )
    if end_mode != "open":
        raise ValueError("Phase 2 GLYCAM loading currently supports only open chains.")

    cache_entry_dir: Path | None = None
    cache_identity: dict[str, object] | None = None
    if cache_enabled:
        if cache_dir is not None or work_dir is None:
            cache_entry_dir, cache_identity = glycam_cache_dir(
                cache_dir,
                polymer=polymer,
                representation=representation,
                end_mode=end_mode,
            )
        else:
            cache_entry_dir = Path(work_dir)
            _, cache_identity = glycam_cache_identity(
                polymer=polymer,
                representation=representation,
                end_mode=end_mode,
            )

    memory_cache_key: tuple[str, str, str, str | None] | None = None
    if cache_enabled and cache_entry_dir is not None:
        memory_cache_key = (
            str(polymer),
            str(representation),
            str(end_mode),
            str(cache_entry_dir.resolve()),
        )
    elif not cache_enabled and work_dir is None:
        memory_cache_key = (str(polymer), str(representation), str(end_mode), None)

    if memory_cache_key is not None:
        cached = _GLYCAM_PARAMS_CACHE.get(memory_cache_key)
        if cached is not None:
            return _with_cache_provenance(
                cached,
                cache_enabled=cache_enabled,
                cache_entry_dir=cache_entry_dir,
                cache_hit=True,
                cache_kind="memory",
            )

    if cache_enabled and cache_entry_dir is not None:
        cached = load_cached_glycam_params(cache_entry_dir)
        if cached is not None:
            if memory_cache_key is not None:
                _GLYCAM_PARAMS_CACHE[memory_cache_key] = cached
            return _with_cache_provenance(
                cached,
                cache_enabled=True,
                cache_entry_dir=cache_entry_dir,
                cache_hit=True,
                cache_kind="disk",
            )

    _ensure_required_tools(("tleap",))

    atom_params: dict[tuple[ResidueRole, str], GlycamAtomParams] = {}
    residue_terms: dict[ResidueRole, dict[str, object]] = {}
    linkage_terms: dict[tuple[ResidueRole, ResidueRole], dict[str, object]] = {}
    reference_runs: list[dict[str, object]] = []

    base_dir = (
        cache_entry_dir
        if cache_enabled and cache_entry_dir is not None
        else (None if work_dir is None else Path(work_dir))
    )
    for reference_dp in (2, 4):
        ref_dir = None if base_dir is None else base_dir / f"dp{reference_dp}"
        reference = _build_reference_prmtop(
            polymer=polymer,
            dp=reference_dp,
            work_dir=ref_dir,
        )
        reference_runs.append(reference)
        _partition_reference_terms(
            prmtop_path=reference["prmtop"],
            dp=reference_dp,
            atom_params=atom_params,
            residue_terms=residue_terms,
            linkage_terms=linkage_terms,
        )

    residue_templates = {
        role: GlycamResidueTemplate(
            residue_role=role,
            residue_name=str(data["residue_name"]),
            atom_names=tuple(sorted(data["atom_names"])),
            bonds=tuple(sorted(data["bonds"].values(), key=lambda item: item.atoms)),
            angles=tuple(sorted(data["angles"].values(), key=lambda item: item.atoms)),
            torsions=_sorted_torsions(data["torsions"]),
        )
        for role, data in residue_terms.items()
    }
    linkage_templates = {
        roles: GlycamLinkageTemplate(
            residue_roles=roles,
            bonds=tuple(sorted(data["bonds"].values(), key=lambda item: item.atoms)),
            angles=tuple(sorted(data["angles"].values(), key=lambda item: item.atoms)),
            torsions=_sorted_torsions(data["torsions"]),
        )
        for roles, data in linkage_terms.items()
    }
    _validate_extracted_templates(residue_templates, linkage_templates)

    params = GlycamParams(
        polymer=polymer,
        representation=representation,
        end_mode=end_mode,
        atom_params=atom_params,
        residue_templates=residue_templates,
        linkage_templates=linkage_templates,
        supported_states=tuple(
            (str(polymer), str(representation), str(end_mode), str(role))
            for role in (
                "terminal_reducing",
                "internal",
                "terminal_nonreducing",
            )
        ),
        provenance={
            "parameter_backend": "glycam_reference_extract",
            "reference_runs": [
                {
                    "reference_dp": int(run["reference_dp"]),
                    "sequence": list(run["sequence"]),
                    "prmtop": str(run["prmtop"]),
                    "inpcrd": str(run["inpcrd"]),
                    "tleap_input": str(run["tleap_input"]),
                    "tleap_log": str(run["tleap_log"]),
                }
                for run in reference_runs
            ],
        },
    )
    if cache_enabled and cache_entry_dir is not None and cache_identity is not None:
        store_cached_glycam_params(
            cache_entry_dir,
            identity=cache_identity,
            params=params,
        )
    if memory_cache_key is not None:
        _GLYCAM_PARAMS_CACHE[memory_cache_key] = params
    return _with_cache_provenance(
        params,
        cache_enabled=cache_enabled,
        cache_entry_dir=cache_entry_dir,
        cache_hit=False,
        cache_kind="build" if cache_enabled else "disabled",
    )
