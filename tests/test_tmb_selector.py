# tests/test_tmb_selector.py
"""Verify TMB selector template and attachment."""
from __future__ import annotations

import numpy as np

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.functionalization import attach_selector
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.chemistry.selector_library.tmb import make_tmb_template
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
    tmb = make_tmb_template()
    assert tmb.name == "tmb"
    assert tmb.mol.GetNumAtoms() > 0
    assert tmb.mol.GetNumConformers() == 1
    assert tmb.attach_atom_idx is not None
    assert tmb.attach_dummy_idx is not None
    assert tmb.attach_atom_idx != tmb.attach_dummy_idx
    # TMB is an ester: no donors, one acceptor (carbonyl O)
    assert len(tmb.donors) == 0
    assert len(tmb.acceptors) == 1


def test_tmb_attach_at_c6() -> None:
    template = make_glucose_template("amylose")
    tmb = make_tmb_template()
    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    n_before = mol.GetNumAtoms()
    mol = attach_selector(
        mol_polymer=mol, template=template,
        residue_index=0, site="C6", selector=tmb,
        linkage_type="ester",
    )

    added = tmb.mol.GetNumAtoms() - 1  # dummy removed
    assert mol.GetNumAtoms() == n_before + added
    assert mol.GetNumConformers() == 1


def test_tmb_sanitizes() -> None:
    template = make_glucose_template("amylose")
    tmb = make_tmb_template()
    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol, template=template,
            residue_index=i, site="C6", selector=tmb,
            linkage_type="ester",
        )

    from rdkit import Chem
    problems = Chem.DetectChemistryProblems(mol)
    assert len(problems) == 0
