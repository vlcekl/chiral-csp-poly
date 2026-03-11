from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from poly_csp.structure.local_frames import compute_residue_local_frame
from poly_csp.structure.periodic_handoff import (
    PeriodicHandoffCleanupSpec,
    PeriodicHandoffSpec,
    build_open_handoff_receptor,
    extract_periodic_handoff_template,
    run_open_handoff_cleanup_relaxation,
)
from poly_csp.forcefield.relaxation import RelaxSpec
from poly_csp.topology.utils import residue_label_maps
from poly_csp.topology.selectors import SelectorRegistry
from tests.support import build_forcefield_mol, test_helix as _helix


_FRAME_LABELS = ("C1", "C2", "C3", "C4", "O4")
_HELIX_CORE_BACKBONE_ATOM_NAMES = {"C1", "C2", "C3", "C4", "C5", "O4", "O5"}


def _residue_frame(mol, residue_index: int) -> tuple[np.ndarray, np.ndarray]:
    maps = residue_label_maps(mol)
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    coords_res = np.array([xyz[int(maps[residue_index][label])] for label in _FRAME_LABELS], dtype=float)
    labels = {label: idx for idx, label in enumerate(_FRAME_LABELS)}
    return compute_residue_local_frame(coords_res, labels)


def test_extract_periodic_handoff_template_requires_periodic_forcefield_molecule() -> None:
    mol = build_forcefield_mol(polymer="amylose", dp=2, end_mode="open")

    with pytest.raises(ValueError, match="end_mode='periodic'"):
        extract_periodic_handoff_template(mol)


def test_extract_periodic_handoff_template_requires_periodic_box_vectors() -> None:
    mol = build_forcefield_mol(polymer="amylose", dp=4, end_mode="periodic")
    for prop in ("_poly_csp_box_a_A", "_poly_csp_box_b_A", "_poly_csp_box_c_A"):
        mol.ClearProp(prop)

    with pytest.raises(ValueError, match="stored periodic box vectors"):
        extract_periodic_handoff_template(mol)


def test_extract_periodic_handoff_template_collects_transferable_atoms() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )

    result = extract_periodic_handoff_template(mol)
    template = result.template

    assert template.unit_cell_dp == 4
    assert template.selector_sites == ("C6",)
    assert len(template.residue_classes) == 4
    assert result.extracted_atom_count > 0
    assert result.extracted_backbone_atom_count > 0
    assert result.extracted_selector_atom_count > 0
    assert result.extracted_connector_atom_count > 0

    for residue_class in template.residue_classes:
        seen_keys = set()
        assert residue_class.component_counts["backbone"] > 0
        assert residue_class.component_counts["selector"] > 0
        assert residue_class.component_counts["connector"] > 0
        for atom_geom in residue_class.atom_geometries:
            key = atom_geom.key
            assert key not in seen_keys
            seen_keys.add(key)
            if key.component == "backbone":
                assert key.atom_name not in _HELIX_CORE_BACKBONE_ATOM_NAMES
                assert key.parent_atom_name not in _HELIX_CORE_BACKBONE_ATOM_NAMES
            else:
                assert key.site == "C6"
                assert key.selector_local_idx is not None


def test_extract_periodic_handoff_template_respects_component_filter_spec() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )

    result = extract_periodic_handoff_template(
        mol,
        spec=PeriodicHandoffSpec(
            include_backbone_exocyclic=False,
            include_connector=False,
            include_selector=True,
        ),
    )

    assert result.extracted_backbone_atom_count == 0
    assert result.extracted_connector_atom_count == 0
    assert result.extracted_selector_atom_count > 0
    for residue_class in result.template.residue_classes:
        assert set(residue_class.component_counts) == {"selector"}
        assert residue_class.component_counts["selector"] > 0


def test_extract_periodic_handoff_template_fails_on_missing_manifest_source() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )
    broken_atom = next(
        atom
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "selector"
    )
    broken_atom.ClearProp("_poly_csp_manifest_source")

    with pytest.raises(ValueError, match="_poly_csp_manifest_source"):
        extract_periodic_handoff_template(mol)


def test_extract_periodic_handoff_template_local_coords_roundtrip_source_geometry() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))

    result = extract_periodic_handoff_template(mol)
    for residue_class in result.template.residue_classes:
        frame_r, frame_t = _residue_frame(mol, residue_class.residue_index)
        for atom_geom in residue_class.atom_geometries:
            local = np.asarray(atom_geom.local_coords_A, dtype=float)
            rebuilt = local @ frame_r.T + frame_t
            expected = xyz[int(atom_geom.global_atom_index)]
            assert np.allclose(rebuilt, expected, atol=1e-6)


def test_build_open_handoff_receptor_requires_odd_n_cells() -> None:
    selector = SelectorRegistry.get("35dmpc")
    periodic = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )
    template = extract_periodic_handoff_template(periodic).template

    with pytest.raises(ValueError, match="odd n_cells"):
        build_open_handoff_receptor(
            periodic,
            template,
            _helix(),
            n_cells=4,
        )


def test_build_open_handoff_receptor_places_periodic_geometry_on_central_cell() -> None:
    selector = SelectorRegistry.get("35dmpc")
    periodic = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )
    extracted = extract_periodic_handoff_template(periodic)

    result = build_open_handoff_receptor(
        periodic,
        extracted.template,
        _helix(),
        n_cells=3,
    )

    expected_transferred = sum(
        len(residue_class.atom_geometries)
        for residue_class in extracted.template.residue_classes
    ) * 3
    assert result.mol.GetProp("_poly_csp_end_mode") == "open"
    assert int(result.mol.GetIntProp("_poly_csp_dp")) == 12
    assert result.expanded_dp == 12
    assert result.n_cells == 3
    assert result.selector_name == "35dmpc"
    assert result.selector_sites == ("C6",)
    assert result.interior_residue_indices == (4, 5, 6, 7)
    assert result.transferred_atom_count == expected_transferred
    assert result.transfer_rmsd_A < 1e-8
    assert result.transfer_max_deviation_A < 1e-8
    assert result.interior_transferred_atom_count == sum(
        len(extracted.template.residue_classes[residue_index].atom_geometries)
        for residue_index in range(extracted.template.unit_cell_dp)
    )
    assert result.interior_transfer_rmsd_A < 1e-8
    assert result.interior_transfer_max_deviation_A < 1e-8


def test_run_open_handoff_cleanup_relaxation_builds_interior_and_terminal_reference_groups(
    monkeypatch,
) -> None:
    selector = SelectorRegistry.get("35dmpc")
    periodic = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )
    extracted = extract_periodic_handoff_template(periodic)
    handoff = build_open_handoff_receptor(
        periodic,
        extracted.template,
        _helix(),
        n_cells=3,
    )
    calls: dict[str, object] = {}

    def fake_run_staged_relaxation(*, mol, spec, selector=None, extra_positional_restraints=(), **kwargs):
        calls["mol"] = mol
        calls["spec"] = spec
        calls["selector"] = selector
        calls["extra_positional_restraints"] = extra_positional_restraints
        return Chem.Mol(mol), {"enabled": True, "protocol": "fake"}

    monkeypatch.setattr(
        "poly_csp.structure.periodic_handoff._run_staged_relaxation",
        fake_run_staged_relaxation,
    )

    relaxed, summary = run_open_handoff_cleanup_relaxation(
        handoff,
        extracted.template,
        RelaxSpec(
            enabled=True,
            positional_k=5000.0,
            dihedral_k=500.0,
            hbond_k=50.0,
            soft_n_stages=1,
            soft_max_iterations=5,
            full_max_iterations=5,
            final_restraint_factor=0.2,
            freeze_backbone=True,
            anneal_enabled=False,
        ),
        cleanup_spec=PeriodicHandoffCleanupSpec(
            enabled=True,
            interior_positional_k=1000.0,
            terminal_positional_k=250.0,
        ),
    )

    groups = calls["extra_positional_restraints"]
    assert calls["mol"] is handoff.mol
    assert calls["selector"] is not None
    assert len(groups) == 2
    assert groups[0].label == "interior"
    assert groups[0].parameter_name == "k_pos_handoff_interior"
    assert groups[0].k_kj_per_mol_nm2 == 1000.0
    assert len(groups[0].atom_indices) == handoff.interior_transferred_atom_count
    assert groups[1].label == "terminal"
    assert groups[1].parameter_name == "k_pos_handoff_terminal"
    assert groups[1].k_kj_per_mol_nm2 == 250.0
    assert len(groups[1].atom_indices) == (
        handoff.transferred_atom_count - handoff.interior_transferred_atom_count
    )
    assert relaxed.transfer_rmsd_A < 1e-8
    assert relaxed.interior_transfer_rmsd_A < 1e-8
    assert summary["periodic_handoff_cleanup"]["enabled"] is True
    assert len(summary["periodic_handoff_cleanup"]["restraint_groups"]) == 2
