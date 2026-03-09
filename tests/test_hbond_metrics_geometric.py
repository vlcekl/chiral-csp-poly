from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.ordering.hbonds import compute_hbond_metrics
from poly_csp.structure.pbc import set_box_vectors


def _selector_test_mol() -> Chem.Mol:
    rw = Chem.RWMol()
    n_idx = rw.AddAtom(Chem.Atom(7))
    c_donor_idx = rw.AddAtom(Chem.Atom(6))
    o_idx = rw.AddAtom(Chem.Atom(8))
    c_acceptor_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(n_idx, c_donor_idx, Chem.BondType.SINGLE)
    rw.AddBond(o_idx, c_acceptor_idx, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _annotate_selector_atoms(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetIntProp("_poly_csp_selector_instance", 1)
        atom.SetIntProp("_poly_csp_residue_index", 0)
        atom.SetProp("_poly_csp_site", "C6")
        atom.SetIntProp("_poly_csp_selector_local_idx", int(atom.GetIdx()))
    return out


def _set_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def _selector_template() -> SelectorTemplate:
    mol = _selector_test_mol()
    return SelectorTemplate(
        name="test_selector",
        mol=mol,
        attach_atom_idx=1,
        dihedrals={},
        donors=(0,),
        acceptors=(2,),
    )


def test_hbond_metrics_distance_pass_geometric_fail() -> None:
    selector = _selector_template()
    mol = _annotate_selector_atoms(selector.mol)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # donor N
            [1.0, 0.0, 0.0],  # donor proxy C -> donor angle ~90
            [0.0, 0.0, 2.8],  # acceptor O (within distance threshold)
            [1.0, 0.0, 2.8],  # acceptor proxy C -> acceptor angle ~90
        ],
        dtype=float,
    )
    mol = _set_coords(mol, coords)

    metrics = compute_hbond_metrics(
        mol=mol,
        selector=selector,
        max_distance_A=3.2,
        min_donor_angle_deg=110.0,
        min_acceptor_angle_deg=120.0,
    )
    assert metrics.total_pairs == 1
    assert metrics.donor_count == 1
    assert metrics.like_satisfied_pairs == 1
    assert metrics.geometric_satisfied_pairs == 0
    assert metrics.like_satisfied_donors == 1
    assert metrics.geometric_satisfied_donors == 0
    assert metrics.like_fraction == 1.0
    assert metrics.geometric_fraction == 0.0
    assert metrics.like_donor_occupancy_fraction == 1.0
    assert metrics.geometric_donor_occupancy_fraction == 0.0


def test_hbond_metrics_geometric_pass() -> None:
    selector = _selector_template()
    mol = _annotate_selector_atoms(selector.mol)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],   # donor N
            [0.0, 0.0, 1.0],   # donor proxy C -> donor angle 180
            [0.0, 0.0, 2.8],   # acceptor O
            [0.0, 0.0, 3.8],   # acceptor proxy C -> acceptor angle 180
        ],
        dtype=float,
    )
    mol = _set_coords(mol, coords)

    metrics = compute_hbond_metrics(
        mol=mol,
        selector=selector,
        max_distance_A=3.2,
        min_donor_angle_deg=100.0,
        min_acceptor_angle_deg=90.0,
    )
    assert metrics.total_pairs == 1
    assert metrics.donor_count == 1
    assert metrics.like_satisfied_pairs == 1
    assert metrics.geometric_satisfied_pairs == 1
    assert metrics.like_satisfied_donors == 1
    assert metrics.geometric_satisfied_donors == 1
    assert metrics.like_fraction == 1.0
    assert metrics.geometric_fraction == 1.0
    assert metrics.like_donor_occupancy_fraction == 1.0
    assert metrics.geometric_donor_occupancy_fraction == 1.0


def test_hbond_metrics_no_conformer_returns_zeroes() -> None:
    selector = _selector_template()
    mol = _annotate_selector_atoms(selector.mol)
    mol.RemoveAllConformers()
    metrics = compute_hbond_metrics(mol=mol, selector=selector)
    assert metrics.total_pairs == 0
    assert metrics.donor_count == 0
    assert metrics.like_satisfied_pairs == 0
    assert metrics.geometric_satisfied_pairs == 0
    assert metrics.like_satisfied_donors == 0
    assert metrics.geometric_satisfied_donors == 0
    assert metrics.like_fraction == 0.0
    assert metrics.geometric_fraction == 0.0
    assert metrics.like_donor_occupancy_fraction == 0.0
    assert metrics.geometric_donor_occupancy_fraction == 0.0


def test_hbond_metrics_periodic_wrap_detects_last_first_neighbor_pair() -> None:
    rw = Chem.RWMol()
    n_idx = rw.AddAtom(Chem.Atom(7))
    c_donor_idx = rw.AddAtom(Chem.Atom(6))
    o_idx = rw.AddAtom(Chem.Atom(8))
    c_acceptor_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(n_idx, c_donor_idx, Chem.BondType.SINGLE)
    rw.AddBond(o_idx, c_acceptor_idx, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    donor = mol.GetAtomWithIdx(n_idx)
    donor.SetIntProp("_poly_csp_selector_instance", 1)
    donor.SetIntProp("_poly_csp_residue_index", 3)
    donor.SetProp("_poly_csp_site", "C6")
    donor.SetIntProp("_poly_csp_selector_local_idx", 0)

    donor_proxy = mol.GetAtomWithIdx(c_donor_idx)
    donor_proxy.SetIntProp("_poly_csp_selector_instance", 1)
    donor_proxy.SetIntProp("_poly_csp_residue_index", 3)
    donor_proxy.SetProp("_poly_csp_site", "C6")
    donor_proxy.SetIntProp("_poly_csp_selector_local_idx", 1)

    acceptor = mol.GetAtomWithIdx(o_idx)
    acceptor.SetIntProp("_poly_csp_selector_instance", 2)
    acceptor.SetIntProp("_poly_csp_residue_index", 0)
    acceptor.SetProp("_poly_csp_site", "C6")
    acceptor.SetIntProp("_poly_csp_selector_local_idx", 2)

    acceptor_proxy = mol.GetAtomWithIdx(c_acceptor_idx)
    acceptor_proxy.SetIntProp("_poly_csp_selector_instance", 2)
    acceptor_proxy.SetIntProp("_poly_csp_residue_index", 0)
    acceptor_proxy.SetProp("_poly_csp_site", "C6")
    acceptor_proxy.SetIntProp("_poly_csp_selector_local_idx", 3)

    mol.SetProp("_poly_csp_end_mode", "periodic")
    mol.SetIntProp("_poly_csp_dp", 4)
    set_box_vectors(mol, 10.0, 10.0, 10.0)

    coords = np.array(
        [
            [9.4, 0.0, 0.0],
            [8.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.8, 0.0, 0.0],
        ],
        dtype=float,
    )
    mol = _set_coords(mol, coords)

    metrics = compute_hbond_metrics(
        mol=mol,
        selector=_selector_template(),
        max_distance_A=2.0,
        neighbor_window=1,
        min_donor_angle_deg=0.0,
        min_acceptor_angle_deg=0.0,
    )

    assert metrics.total_pairs == 1
    assert metrics.donor_count == 1
    assert metrics.like_satisfied_pairs == 1
    assert metrics.geometric_satisfied_pairs == 1
    assert metrics.like_satisfied_donors == 1
    assert metrics.geometric_satisfied_donors == 1
    assert metrics.mean_like_distance_A == pytest.approx(1.4)
    assert metrics.like_donor_occupancy_fraction == 1.0
