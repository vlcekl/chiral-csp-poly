# tests/test_tmb_selector.py
"""Verify TMB selector template and attachment."""
from __future__ import annotations

import numpy as np

from tests.support import build_backbone_coords
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.topology.selectors import SelectorRegistry
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


def test_tmb_template_structure() -> None:
    tmb = SelectorRegistry.get("tmb")
    assert tmb.name == "tmb"
    assert tmb.mol.GetNumAtoms() > 0
    assert tmb.mol.GetNumConformers() == 1
    assert tmb.attach_atom_idx is not None
    assert tmb.attach_dummy_idx is not None
    assert tmb.attach_atom_idx != tmb.attach_dummy_idx
    assert tmb.linkage_type == "ester"
    # TMB is an ester: no donors, one acceptor (carbonyl O)
    assert len(tmb.donors) == 0
    assert len(tmb.acceptors) == 1
    assert set(tmb.connector_local_roles.values()) == {"carbonyl_c", "carbonyl_o"}
    for atom_idx in tmb.connector_local_roles:
        atom = tmb.mol.GetAtomWithIdx(atom_idx)
        assert atom.GetTotalNumHs(includeNeighbors=True) == 0


def test_tmb_attach_at_c6() -> None:
    template = make_glucose_template("amylose")
    tmb = SelectorRegistry.get("tmb")
    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    n_before = mol.GetNumAtoms()
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0, site="C6", selector=tmb,
        linkage_type="ester",
    )

    added = tmb.mol.GetNumAtoms() - 1  # dummy removed
    assert mol.GetNumAtoms() == n_before + added
    assert mol.GetNumConformers() == 1


def test_tmb_sanitizes() -> None:
    template = make_glucose_template("amylose")
    tmb = SelectorRegistry.get("tmb")
    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol,
            residue_index=i, site="C6", selector=tmb,
            linkage_type="ester",
        )

    from rdkit import Chem
    problems = Chem.DetectChemistryProblems(mol)
    assert len(problems) == 0
