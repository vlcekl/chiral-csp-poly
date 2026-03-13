from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.ordering.hbonds import build_hbond_restraint_pairs
from poly_csp.topology.selectors import SelectorTemplate


def _selector_fragment() -> Chem.Mol:
    rw = Chem.RWMol()
    n_idx = rw.AddAtom(Chem.Atom(7))
    c_donor_idx = rw.AddAtom(Chem.Atom(6))
    h_idx = rw.AddAtom(Chem.Atom(1))
    o_idx = rw.AddAtom(Chem.Atom(8))
    c_acceptor_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(n_idx, c_donor_idx, Chem.BondType.SINGLE)
    rw.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)
    rw.AddBond(o_idx, c_acceptor_idx, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _selector_template() -> SelectorTemplate:
    return SelectorTemplate(
        name="test_selector",
        mol=_selector_fragment(),
        attach_atom_idx=1,
        dihedrals={},
        donors=(0,),
        acceptors=(3,),
    )


def _annotated_multimer(residues: list[int]) -> Chem.Mol:
    fragment = _selector_fragment()
    mol = Chem.Mol(fragment)
    for _ in residues[1:]:
        mol = Chem.CombineMols(mol, fragment)
    out = Chem.Mol(mol)
    atoms_per_fragment = fragment.GetNumAtoms()
    for instance_id, residue_index in enumerate(residues, start=1):
        offset = (instance_id - 1) * atoms_per_fragment
        for local_idx in range(atoms_per_fragment):
            atom = out.GetAtomWithIdx(offset + local_idx)
            atom.SetIntProp("_poly_csp_selector_instance", instance_id)
            atom.SetIntProp("_poly_csp_residue_index", residue_index)
            atom.SetProp("_poly_csp_site", "C6")
            atom.SetIntProp("_poly_csp_selector_local_idx", local_idx)
    return out


def _set_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def test_build_hbond_restraint_pairs_uses_donor_hydrogen_when_present() -> None:
    mol = _annotated_multimer([0, 1])
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [19.0, 0.0, 0.0],
            [2.9, 0.0, 0.0],
            [3.9, 0.0, 0.0],
        ],
        dtype=float,
    )
    mol = _set_coords(mol, coords)

    pairs = build_hbond_restraint_pairs(
        mol,
        _selector_template(),
        max_distance_A=3.0,
        atom_mode="hydrogen_if_present",
        ideal_target_nm=0.18,
    )

    assert len(pairs) == 1
    assert pairs[0][0] == 2
    assert pairs[0][1] == 8
    assert pairs[0][2] == pytest.approx(0.18)


def test_build_hbond_restraint_pairs_converts_short_ideal_target_for_donor_heavy() -> None:
    mol = _annotated_multimer([0, 1])
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [19.0, 0.0, 0.0],
            [2.9, 0.0, 0.0],
            [3.9, 0.0, 0.0],
        ],
        dtype=float,
    )
    mol = _set_coords(mol, coords)

    pairs = build_hbond_restraint_pairs(
        mol,
        _selector_template(),
        max_distance_A=3.1,
        atom_mode="donor_heavy",
        ideal_target_nm=0.18,
    )

    assert len(pairs) == 1
    assert pairs[0][0] == 0
    assert pairs[0][1] == 8
    assert pairs[0][2] == pytest.approx(0.28)


def test_build_hbond_restraint_pairs_nearest_unique_limits_each_donor_to_one_acceptor() -> None:
    mol = _annotated_multimer([0, 1, 1])
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [51.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [19.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [41.0, 0.0, 0.0],
            [39.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    mol = _set_coords(mol, coords)

    legacy_pairs = build_hbond_restraint_pairs(
        mol,
        _selector_template(),
        max_distance_A=3.2,
        pairing_mode="legacy_all_pairs",
        atom_mode="hydrogen_if_present",
    )
    unique_pairs = build_hbond_restraint_pairs(
        mol,
        _selector_template(),
        max_distance_A=3.2,
        pairing_mode="nearest_unique",
        atom_mode="hydrogen_if_present",
    )

    assert len(legacy_pairs) == 2
    assert len(unique_pairs) == 1
    assert unique_pairs[0][0] == 2
    assert unique_pairs[0][1] == 8
    assert unique_pairs[0][2] == pytest.approx(0.18)


def test_build_hbond_restraint_pairs_respects_neighbor_window() -> None:
    mol = _annotated_multimer([0, 3])
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [19.0, 0.0, 0.0],
            [2.9, 0.0, 0.0],
            [3.9, 0.0, 0.0],
        ],
        dtype=float,
    )
    mol = _set_coords(mol, coords)
    mol.SetProp("_poly_csp_end_mode", "open")
    mol.SetIntProp("_poly_csp_dp", 4)

    blocked = build_hbond_restraint_pairs(
        mol,
        _selector_template(),
        max_distance_A=3.0,
        neighbor_window=1,
    )
    allowed = build_hbond_restraint_pairs(
        mol,
        _selector_template(),
        max_distance_A=3.0,
        neighbor_window=3,
    )

    assert blocked == []
    assert len(allowed) == 1
    assert allowed[0][0] == 2
    assert allowed[0][1] == 8
    assert allowed[0][2] == pytest.approx(0.19)
