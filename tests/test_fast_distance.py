# tests/test_fast_distance.py
"""Verify cKDTree-based fast scoring matches naive implementation."""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from tests.support import build_backbone_coords
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.structure.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.config.schema import HelixSpec
from poly_csp.ordering.scoring import (
    bonded_exclusion_pairs,
    min_distance_by_class,
    min_distance_by_class_fast,
    min_interatomic_distance,
    min_interatomic_distance_fast,
)
from poly_csp.structure.pbc import get_box_vectors_A
from tests.support import build_forcefield_mol


def _helix() -> HelixSpec:
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


def _heavy_mask(mol: Chem.Mol) -> np.ndarray:
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def _build_mol_with_selectors():
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()
    dp = 4
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol,
            residue_index=i, site="C6", selector=selector,
        )
    return mol


def test_min_distance_fast_matches_naive() -> None:
    mol = _build_mol_with_selectors()
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    heavy = _heavy_mask(mol)
    excluded = bonded_exclusion_pairs(mol, max_path_length=2)

    naive = min_interatomic_distance(xyz, heavy, excluded)
    fast = min_interatomic_distance_fast(xyz, heavy, excluded, cutoff=3.0)
    # If the minimum distance < cutoff, they should agree
    if naive < 3.0:
        assert abs(naive - fast) < 1e-6, f"naive={naive}, fast={fast}"


def test_class_distance_fast_matches_naive() -> None:
    mol = _build_mol_with_selectors()
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    heavy = _heavy_mask(mol)
    excluded = bonded_exclusion_pairs(mol, max_path_length=2)

    naive = min_distance_by_class(mol, xyz, heavy, excluded)
    fast = min_distance_by_class_fast(mol, xyz, heavy, excluded, cutoff=3.0)

    for key in ("backbone_backbone", "backbone_selector", "selector_selector"):
        n_val = naive[key]
        f_val = fast[key]
        if np.isfinite(n_val) and n_val < 3.0:
            assert abs(n_val - f_val) < 1e-6, f"key={key} naive={n_val} fast={f_val}"


def test_periodic_fast_distances_match_naive() -> None:
    selector = make_35_dmpc_template()
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=4,
        selector=selector,
        site="C6",
        end_mode="periodic",
    )
    conf = mol.GetConformer(0)
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if int(atom.GetIntProp("_poly_csp_selector_instance")) != 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        conf.SetAtomPosition(atom.GetIdx(), (float(pos.x + 120.0), float(pos.y), float(pos.z)))

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    heavy = _heavy_mask(mol)
    excluded = bonded_exclusion_pairs(mol, max_path_length=2)
    box_vectors_A = get_box_vectors_A(mol)

    naive = min_interatomic_distance(xyz, heavy, excluded, box_vectors_A=box_vectors_A)
    fast = min_interatomic_distance_fast(
        xyz,
        heavy,
        excluded,
        cutoff=3.0,
        box_vectors_A=box_vectors_A,
    )
    assert abs(naive - fast) < 1e-6, f"naive={naive}, fast={fast}"

    naive_class = min_distance_by_class(
        mol,
        xyz,
        heavy,
        excluded,
        box_vectors_A=box_vectors_A,
    )
    fast_class = min_distance_by_class_fast(
        mol,
        xyz,
        heavy,
        excluded,
        cutoff=3.0,
        box_vectors_A=box_vectors_A,
    )
    for key in ("backbone_backbone", "backbone_selector", "selector_selector"):
        assert abs(naive_class[key] - fast_class[key]) < 1e-6
