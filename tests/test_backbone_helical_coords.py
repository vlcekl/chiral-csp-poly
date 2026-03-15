from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from poly_csp.cache_versions import (
    BACKBONE_POSE_CACHE_SCHEMA_VERSION,
    BACKBONE_POSE_MODEL_VERSION,
)
from tests.support import build_backbone_coords
from poly_csp.config.schema import HelixSpec
import poly_csp.structure.backbone_builder as backbone_builder_mod
from poly_csp.structure.dihedrals import measure_dihedral_rad
from poly_csp.structure.matrix import ScrewTransform
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.terminals import apply_terminal_mode


_ROOT = Path(__file__).resolve().parents[1]
_HELIX_DIR = _ROOT / "conf" / "structure" / "helix"


def _test_helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0 * np.pi / 2.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def _load_helix_preset(name: str) -> HelixSpec:
    payload = OmegaConf.to_container(
        OmegaConf.load(_HELIX_DIR / f"{name}.yaml"),
        resolve=True,
    )
    assert isinstance(payload, dict)
    return HelixSpec.model_validate(payload)


def test_build_backbone_coords_shape_and_symmetry() -> None:
    template = make_glucose_template("amylose")
    helix = _test_helix()
    dp = 6

    coords = build_backbone_coords(template=template, helix=helix, dp=dp)
    n = template.mol.GetNumAtoms()
    assert coords.shape == (dp * n, 3)

    screw = ScrewTransform(theta_rad=helix.theta_rad, rise_A=helix.rise_A)
    res0 = coords[:n]
    for i in range(dp):
        resi = coords[i * n : (i + 1) * n]
        pred = screw.apply(res0, i)
        rmsd = np.sqrt(np.mean(np.sum((resi - pred) ** 2, axis=1)))
        assert rmsd < 1e-9


def test_ring_centroid_radius_is_constant_across_residues() -> None:
    template = make_glucose_template("amylose")
    helix = _test_helix()
    dp = 8

    coords = build_backbone_coords(template=template, helix=helix, dp=dp)
    n = template.mol.GetNumAtoms()
    ring_idx = [
        template.atom_idx["C1"],
        template.atom_idx["C2"],
        template.atom_idx["C3"],
        template.atom_idx["C4"],
        template.atom_idx["C5"],
        template.atom_idx["O5"],
    ]

    radii = []
    for i in range(dp):
        block = coords[i * n : (i + 1) * n]
        centroid = block[ring_idx].mean(axis=0)
        radii.append(float(np.linalg.norm(centroid[:2])))

    assert max(radii) - min(radii) < 1e-9


def test_derivatized_amylose_preset_normalizes_expected_geometry() -> None:
    helix = _load_helix_preset("amylose_4_3_derivatized")

    assert helix.name == "amylose_CSP_4_3_derivatized"
    assert helix.repeat_residues == 4
    assert helix.repeat_turns == 3
    assert helix.rise_A == pytest.approx(15.614 / 4.0)
    assert helix.axial_repeat_A == pytest.approx(15.614)
    assert helix.pitch_A == pytest.approx(15.614 / 3.0)
    assert helix.glycosidic_phi_deg == pytest.approx(-68.5)
    assert helix.glycosidic_psi_deg == pytest.approx(-42.0)


def test_derivatized_cellulose_preset_normalizes_expected_geometry() -> None:
    helix = _load_helix_preset("cellulose_3_2_derivatized")

    assert helix.name == "cellulose_CSP_3_2_derivatized"
    assert helix.repeat_residues == 3
    assert helix.repeat_turns == 2
    assert helix.rise_A == pytest.approx(15.3 / 3.0)
    assert helix.axial_repeat_A == pytest.approx(15.3)
    assert helix.pitch_A == pytest.approx(15.3 / 2.0)
    assert helix.glycosidic_phi_deg == pytest.approx(60.0)
    assert helix.glycosidic_psi_deg == pytest.approx(0.0)


def test_derivatized_amylose_backbone_structure_biases_toward_glycosidic_targets() -> None:
    helix = _load_helix_preset("amylose_4_3_derivatized")
    helix_without_targets = helix.model_copy(
        update={
            "glycosidic_phi_deg": None,
            "glycosidic_psi_deg": None,
        }
    )
    template = make_glucose_template("amylose", monomer_representation="anhydro")
    topology = polymerize(
        template=template,
        dp=4,
        linkage="1-4",
        anomer="alpha",
    )
    topology = apply_terminal_mode(
        mol=topology,
        mode="periodic",
        caps={},
        representation="anhydro",
    )

    baseline = backbone_builder_mod.build_backbone_structure(topology, helix_without_targets)
    result = backbone_builder_mod.build_backbone_structure(topology, helix)

    def _measured_dihedrals(build_result):
        maps = build_result.residue_maps
        mol = build_result.mol
        coords = np.asarray(
            mol.GetConformer(0).GetPositions(),
            dtype=float,
        ).reshape((-1, 3))
        h1_idx = next(
            int(atom.GetIdx())
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_residue_index")
            and int(atom.GetIntProp("_poly_csp_residue_index")) == 0
            and atom.HasProp("_poly_csp_atom_name")
            and atom.GetProp("_poly_csp_atom_name") == "H1"
        )
        h4_idx = next(
            int(atom.GetIdx())
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_residue_index")
            and int(atom.GetIntProp("_poly_csp_residue_index")) == 1
            and atom.HasProp("_poly_csp_atom_name")
            and atom.GetProp("_poly_csp_atom_name") == "H4"
        )
        phi_deg = float(
            np.rad2deg(
                measure_dihedral_rad(
                    coords,
                    h1_idx,
                    maps[0]["C1"],
                    maps[1]["O4"],
                    maps[1]["C4"],
                )
            )
        )
        psi_deg = float(
            np.rad2deg(
                measure_dihedral_rad(
                    coords,
                    maps[0]["C1"],
                    maps[1]["O4"],
                    maps[1]["C4"],
                    h4_idx,
                )
            )
        )
        return phi_deg, psi_deg

    baseline_phi_deg, baseline_psi_deg = _measured_dihedrals(baseline)
    phi_deg, psi_deg = _measured_dihedrals(result)

    baseline_error = abs(baseline_phi_deg - float(helix.glycosidic_phi_deg)) + abs(
        baseline_psi_deg - float(helix.glycosidic_psi_deg)
    )
    targeted_error = abs(phi_deg - float(helix.glycosidic_phi_deg)) + abs(
        psi_deg - float(helix.glycosidic_psi_deg)
    )

    assert result.mol.GetDoubleProp("_poly_csp_helix_glycosidic_phi_deg") == pytest.approx(
        float(helix.glycosidic_phi_deg)
    )
    assert result.mol.GetDoubleProp("_poly_csp_helix_glycosidic_psi_deg") == pytest.approx(
        float(helix.glycosidic_psi_deg)
    )
    assert targeted_error < baseline_error


def _open_backbone_topology(polymer: str = "amylose"):
    template = make_glucose_template(polymer, monomer_representation="anhydro")
    topology = polymerize(
        template=template,
        dp=2,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    return apply_terminal_mode(
        mol=topology,
        mode="open",
        caps={},
        representation="anhydro",
    )


def test_backbone_pose_disk_cache_reuses_saved_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = make_glucose_template("amylose")
    helix = _test_helix()
    cache_dir = tmp_path / "backbone_pose_cache"

    monkeypatch.setattr(backbone_builder_mod, "_BACKBONE_POSE_CACHE_DIR", cache_dir)
    backbone_builder_mod._BACKBONE_POSE_CACHE.clear()

    coords_first = backbone_builder_mod.build_backbone_heavy_coords(template, helix, 6)
    assert list(cache_dir.rglob("pose.json"))

    backbone_builder_mod._BACKBONE_POSE_CACHE.clear()

    def _raise_if_recomputed():
        raise AssertionError("Backbone pose should have been loaded from disk cache.")

    monkeypatch.setattr(backbone_builder_mod, "_candidate_backbone_poses", _raise_if_recomputed)
    coords_second = backbone_builder_mod.build_backbone_heavy_coords(template, helix, 6)

    assert np.allclose(coords_first, coords_second)


def test_backbone_pose_cache_summary_reports_build_memory_and_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _open_backbone_topology("amylose")
    helix = _test_helix()
    cache_dir = tmp_path / "backbone_pose_cache"

    monkeypatch.setattr(backbone_builder_mod, "_BACKBONE_POSE_CACHE_DIR", cache_dir)
    backbone_builder_mod._BACKBONE_POSE_CACHE.clear()

    first = backbone_builder_mod.build_backbone_structure(topology, helix)
    second = backbone_builder_mod.build_backbone_structure(topology, helix)
    backbone_builder_mod._BACKBONE_POSE_CACHE.clear()
    third = backbone_builder_mod.build_backbone_structure(topology, helix)

    assert first.pose_cache_summary.kind == "build"
    assert first.pose_cache_summary.hit is False
    assert first.pose_cache_summary.persisted is True
    assert first.pose_cache_summary.schema_version == BACKBONE_POSE_CACHE_SCHEMA_VERSION
    assert first.pose_cache_summary.model_version == BACKBONE_POSE_MODEL_VERSION
    assert Path(first.pose_cache_summary.entry_path).exists()
    assert first.pose_cache_summary.cache_dir == str(cache_dir)

    assert second.pose_cache_summary.kind == "memory"
    assert second.pose_cache_summary.hit is True
    assert second.pose_cache_summary.entry_path == first.pose_cache_summary.entry_path

    assert third.pose_cache_summary.kind == "disk"
    assert third.pose_cache_summary.hit is True
    assert third.pose_cache_summary.entry_path == first.pose_cache_summary.entry_path


def test_backbone_pose_cache_rejects_stale_model_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _open_backbone_topology("amylose")
    helix = _test_helix()
    cache_dir = tmp_path / "backbone_pose_cache"

    monkeypatch.setattr(backbone_builder_mod, "_BACKBONE_POSE_CACHE_DIR", cache_dir)
    backbone_builder_mod._BACKBONE_POSE_CACHE.clear()

    first = backbone_builder_mod.build_backbone_structure(topology, helix)
    payload_path = Path(first.pose_cache_summary.entry_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["model_version"] = BACKBONE_POSE_MODEL_VERSION - 1
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    backbone_builder_mod._BACKBONE_POSE_CACHE.clear()
    second = backbone_builder_mod.build_backbone_structure(topology, helix)

    assert second.pose_cache_summary.kind == "build"
    assert second.pose_cache_summary.persisted is True
