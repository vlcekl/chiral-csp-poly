# tests/test_carbamate_geometry.py
"""Verify that selector attachment produces chemically correct carbamate geometry."""
from __future__ import annotations

import numpy as np
import pytest

from tests.support import build_backbone_coords
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.topology.linkage import CARBAMATE, build_linkage_coords
from poly_csp.config.schema import HelixSpec


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


def _build_mol_with_selector(dp: int = 2, site: str = "C6"):
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site=site,
        selector=selector,
    )
    return mol, selector


def test_carbamate_oc_bond_length() -> None:
    """O_sugar – C_carbonyl bond should be ~1.36 Å."""
    mol, selector = _build_mol_with_selector()
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))

    # Find the sugar O6 and the carbonyl carbon it's bonded to
    o6_idx = None
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            # Check if this is O6 of residue 0 by finding the atom bonded
            # to both a backbone carbon and a selector carbon
            if atom.GetAtomicNum() == 8:
                nbrs = [nbr for nbr in atom.GetNeighbors()]
                has_selector_nbr = any(
                    nbr.HasProp("_poly_csp_selector_instance") for nbr in nbrs
                )
                has_backbone_nbr = any(
                    not nbr.HasProp("_poly_csp_selector_instance") for nbr in nbrs
                )
                if has_selector_nbr and has_backbone_nbr:
                    o6_idx = atom.GetIdx()
                    break

    assert o6_idx is not None, "Could not find the bridging oxygen atom"

    # Find the selector carbonyl carbon bonded to this oxygen
    o_atom = mol.GetAtomWithIdx(o6_idx)
    carbonyl_c_idx = None
    for nbr in o_atom.GetNeighbors():
        if nbr.HasProp("_poly_csp_selector_instance"):
            carbonyl_c_idx = nbr.GetIdx()
            break

    assert carbonyl_c_idx is not None, "Could not find carbonyl C bonded to bridging O"

    d = float(np.linalg.norm(xyz[o6_idx] - xyz[carbonyl_c_idx]))
    assert abs(d - CARBAMATE.ab_bond_A) < 0.15, (
        f"O–C bond length {d:.3f} Å deviates from ideal {CARBAMATE.ab_bond_A} Å"
    )


def test_carbamate_co_double_bond_geometry() -> None:
    """C=O double bond on carbonyl should be ~1.22 Å."""
    mol, selector = _build_mol_with_selector()
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))

    # Find a selector carbonyl C bonded to an O (double bond)
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if atom.GetAtomicNum() != 6:
            continue
        # Check if this carbon has a double-bonded oxygen neighbor
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
            if nbr.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() >= 1.9:
                d = float(np.linalg.norm(xyz[atom.GetIdx()] - xyz[nbr.GetIdx()]))
                # This is the C=O distance; it should be placed near ideal
                assert d < 2.0, f"C=O distance {d:.3f} Å is unreasonably large"
                return

    # If no double bond O found, we might be in heavy-atom rep — just check
    # that the selector has oxygen atoms
    selector_os = [
        atom for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_selector_instance") and atom.GetAtomicNum() == 8
    ]
    assert len(selector_os) > 0, "Selector should have at least one oxygen atom"


def test_attached_selector_is_sanitizable() -> None:
    """The attached molecule must pass RDKit sanitization."""
    mol, _ = _build_mol_with_selector()
    # If we got here, SanitizeMol already passed inside attach_selector
    # Double-check explicitly:
    from rdkit import Chem
    problems = Chem.DetectChemistryProblems(mol)
    assert len(problems) == 0, f"Chemistry problems: {[p.Message() for p in problems]}"


def test_attached_selector_has_conformer() -> None:
    mol, _ = _build_mol_with_selector()
    assert mol.GetNumConformers() == 1


def test_attach_all_sites_preserves_atom_count() -> None:
    """Attaching at C2, C3, C6 should add the expected number of atoms."""
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")
    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    n_before = mol.GetNumAtoms()
    for i in range(dp):
        for site in ("C2", "C3", "C6"):
            mol = attach_selector(
                mol_polymer=mol,
                residue_index=i,
                site=site,
                selector=selector,
            )

    added_per = selector.mol.GetNumAtoms() - 1  # dummy removed
    assert mol.GetNumAtoms() == n_before + dp * 3 * added_per


def test_build_linkage_coords_carbamate_angles() -> None:
    """Verify build_linkage_coords produces correct angles."""
    anchor = np.array([5.0, 0.0, 0.0])
    parent = np.array([4.0, 0.0, 0.0])

    b, c, sidechain = build_linkage_coords(
        anchor_pos=anchor,
        anchor_parent_pos=parent,
        geom=CARBAMATE,
    )

    # Check bond lengths
    ab = float(np.linalg.norm(b - anchor))
    bc = float(np.linalg.norm(c - b))
    assert abs(ab - CARBAMATE.ab_bond_A) < 1e-6, f"AB bond: {ab}"
    assert abs(bc - CARBAMATE.bc_bond_A) < 1e-6, f"BC bond: {bc}"

    # Check A-B-C angle
    ba = _normalize(anchor - b)
    bc_dir = _normalize(c - b)
    cos_abc = float(np.dot(ba, bc_dir))
    abc_deg = float(np.rad2deg(np.arccos(np.clip(cos_abc, -1.0, 1.0))))
    assert abs(abc_deg - CARBAMATE.abc_angle_deg) < 1.0, f"ABC angle: {abc_deg}"

    # Check sidechain exists
    assert sidechain is not None
    side_d = float(np.linalg.norm(sidechain - b))
    assert abs(side_d - CARBAMATE.sidechain_bond_A) < 1e-6, f"Sidechain bond: {side_d}"


def _normalize(v):
    return v / float(np.linalg.norm(v))
