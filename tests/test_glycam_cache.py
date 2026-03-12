from __future__ import annotations

import json
from pathlib import Path

from poly_csp.cache_versions import (
    RUNTIME_PAYLOAD_CACHE_SCHEMA_VERSION,
    RUNTIME_PAYLOAD_MODEL_VERSION,
)
import poly_csp.forcefield.glycam as glycam_mod
from poly_csp.forcefield.payload_cache import glycam_cache_dir, load_cached_glycam_params


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_load_glycam_params_reuses_persistent_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_calls: list[int] = []
    partition_calls: list[int] = []

    monkeypatch.setattr(glycam_mod, "_ensure_required_tools", lambda tools: None)

    def fake_build_reference_prmtop(polymer, dp, end_mode, work_dir=None):
        build_calls.append(int(dp))
        outdir = Path(work_dir) if work_dir is not None else tmp_path / f"dp{dp}"
        outdir.mkdir(parents=True, exist_ok=True)
        prmtop = outdir / f"glycam_ref_dp{dp}.prmtop"
        inpcrd = outdir / f"glycam_ref_dp{dp}.inpcrd"
        tleap_input = outdir / "tleap.in"
        tleap_log = outdir / "tleap.log"
        for artifact in (prmtop, inpcrd, tleap_input, tleap_log):
            _touch(artifact)
        return {
            "prmtop": str(prmtop),
            "inpcrd": str(inpcrd),
            "tleap_input": str(tleap_input),
            "tleap_log": str(tleap_log),
            "reference_dp": int(dp),
            "sequence": glycam_mod.build_glycam_sequence(
                polymer=polymer,
                dp=dp,
                end_mode=end_mode,
            ),
        }

    def fake_partition_reference_terms(
        prmtop_path,
        dp,
        end_mode,
        atom_params,
        residue_terms,
        linkage_terms,
    ):
        partition_calls.append(int(dp))
        roles = glycam_mod.glycam_residue_roles_for_dp(dp, end_mode=end_mode)
        for role in set(roles):
            residue_name = glycam_mod.GLYCAM_RESIDUE_NAMES[("amylose", role)]
            bucket = residue_terms.setdefault(
                role,
                {
                    "residue_name": residue_name,
                    "atom_names": set(),
                    "bonds": {},
                    "angles": {},
                    "torsions": [],
                },
            )
            bucket["atom_names"].update({"C1", "O4", "H2O"})
            atom_params[(role, "C1")] = glycam_mod.GlycamAtomParams(
                charge_e=0.10,
                sigma_nm=0.31,
                epsilon_kj_per_mol=0.20,
                residue_name=residue_name,
                source_atom_name="C1",
            )
            bond = glycam_mod.GlycamBondTemplate(
                atoms=(
                    glycam_mod.GlycamAtomToken(0, "C1"),
                    glycam_mod.GlycamAtomToken(0, "O4"),
                ),
                length_nm=0.14,
                k_kj_per_mol_nm2=1000.0,
            )
            angle = glycam_mod.GlycamAngleTemplate(
                atoms=(
                    glycam_mod.GlycamAtomToken(0, "C1"),
                    glycam_mod.GlycamAtomToken(0, "O4"),
                    glycam_mod.GlycamAtomToken(0, "H2O"),
                ),
                theta0_rad=2.0,
                k_kj_per_mol_rad2=50.0,
            )
            torsion = glycam_mod.GlycamTorsionTemplate(
                atoms=(
                    glycam_mod.GlycamAtomToken(0, "H2O"),
                    glycam_mod.GlycamAtomToken(0, "O4"),
                    glycam_mod.GlycamAtomToken(0, "C1"),
                    glycam_mod.GlycamAtomToken(0, "H2O"),
                ),
                periodicity=3,
                phase_rad=0.0,
                k_kj_per_mol=1.5,
            )
            bucket["bonds"][bond.atoms] = bond
            bucket["angles"][angle.atoms] = angle
            bucket["torsions"][:] = [torsion]

        for left_role, right_role in zip(roles[:-1], roles[1:], strict=True):
            bucket = linkage_terms.setdefault(
                (left_role, right_role),
                {"bonds": {}, "angles": {}, "torsions": []},
            )
            bond = glycam_mod.GlycamBondTemplate(
                atoms=(
                    glycam_mod.GlycamAtomToken(0, "O4"),
                    glycam_mod.GlycamAtomToken(1, "C1"),
                ),
                length_nm=0.143,
                k_kj_per_mol_nm2=1200.0,
            )
            angle = glycam_mod.GlycamAngleTemplate(
                atoms=(
                    glycam_mod.GlycamAtomToken(0, "C1"),
                    glycam_mod.GlycamAtomToken(0, "O4"),
                    glycam_mod.GlycamAtomToken(1, "C1"),
                ),
                theta0_rad=1.9,
                k_kj_per_mol_rad2=60.0,
            )
            torsion = glycam_mod.GlycamTorsionTemplate(
                atoms=(
                    glycam_mod.GlycamAtomToken(0, "H2O"),
                    glycam_mod.GlycamAtomToken(0, "O4"),
                    glycam_mod.GlycamAtomToken(1, "C1"),
                    glycam_mod.GlycamAtomToken(1, "O4"),
                ),
                periodicity=3,
                phase_rad=0.1,
                k_kj_per_mol=0.8,
            )
            bucket["bonds"][bond.atoms] = bond
            bucket["angles"][angle.atoms] = angle
            bucket["torsions"][:] = [torsion]

    monkeypatch.setattr(glycam_mod, "_build_reference_prmtop", fake_build_reference_prmtop)
    monkeypatch.setattr(glycam_mod, "_partition_reference_terms", fake_partition_reference_terms)

    cache_root = tmp_path / "runtime_cache"
    glycam_mod._GLYCAM_PARAMS_CACHE.clear()
    first = glycam_mod.load_glycam_params(
        polymer="amylose",
        representation="anhydro",
        end_mode="open",
        cache_enabled=True,
        cache_dir=cache_root,
    )

    glycam_mod._GLYCAM_PARAMS_CACHE.clear()
    second = glycam_mod.load_glycam_params(
        polymer="amylose",
        representation="anhydro",
        end_mode="open",
        cache_enabled=True,
        cache_dir=cache_root,
    )

    assert build_calls == [2, 4]
    assert partition_calls == [2, 4]
    assert first.atom_params == second.atom_params
    assert first.residue_templates == second.residue_templates
    assert first.linkage_templates == second.linkage_templates
    assert first.provenance["cache"]["hit"] is False
    assert first.provenance["cache"]["kind"] == "build"
    assert (
        first.provenance["cache"]["schema_version"]
        == RUNTIME_PAYLOAD_CACHE_SCHEMA_VERSION
    )
    assert first.provenance["cache"]["model_version"] == RUNTIME_PAYLOAD_MODEL_VERSION
    assert second.provenance["cache"]["hit"] is True
    assert second.provenance["cache"]["kind"] == "disk"

    entry_dir, _ = glycam_cache_dir(
        cache_root,
        polymer="amylose",
        representation="anhydro",
        end_mode="open",
    )
    assert (entry_dir / "payload.json").exists()
    assert (entry_dir / "dp2" / "glycam_ref_dp2.prmtop").exists()
    assert (entry_dir / "dp4" / "glycam_ref_dp4.prmtop").exists()


def test_load_cached_glycam_params_rejects_stale_model_version(tmp_path: Path) -> None:
    entry_dir = tmp_path / "glycam_entry"
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "payload.json").write_text(
        json.dumps(
            {
                "schema_version": RUNTIME_PAYLOAD_CACHE_SCHEMA_VERSION,
                "model_version": RUNTIME_PAYLOAD_MODEL_VERSION - 1,
                "payload_kind": "glycam_backbone",
                "identity": {"kind": "glycam_backbone"},
                "payload": {},
            }
        ),
        encoding="utf-8",
    )

    assert load_cached_glycam_params(entry_dir) is None
