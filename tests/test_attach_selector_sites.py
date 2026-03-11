from __future__ import annotations

import json

import numpy as np
import pytest
from rdkit import Chem

from tests.support import build_backbone_coords
from poly_csp.topology.reactions import attach_selector, residue_atom_global_index
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


def test_attach_selector_c6_adds_expected_atoms_and_bond() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    n_before = mol.GetNumAtoms()
    added_expected = selector.mol.GetNumAtoms() - 1  # dummy removed on attach
    mol2 = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    assert mol2.GetNumAtoms() == n_before + added_expected
    assert mol2.GetNumConformers() == 1

    o6_global = residue_atom_global_index(
        residue_index=0,
        monomer_atom_count=template.mol.GetNumAtoms(),
        local_atom_index=template.site_idx["O6"],
    )
    o6_neighbors = [a.GetIdx() for a in mol2.GetAtomWithIdx(o6_global).GetNeighbors()]
    assert any(idx >= n_before for idx in o6_neighbors)


def test_attach_selector_all_residues_c6_sanitizes() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    dp = 3
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol,
            residue_index=i,
            site="C6",
            selector=selector,
        )

    assert mol.GetNumConformers() == 1


def test_attach_selector_all_sites_all_residues_increases_atom_count() -> None:
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

    added_per_selector = selector.mol.GetNumAtoms() - 1
    assert mol.GetNumAtoms() == n_before + dp * 3 * added_per_selector
    assert mol.GetNumConformers() == 1


def test_attach_selector_consumes_site_oh_and_preserves_carbamate_nh() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = SelectorRegistry.get("35dmpc")

    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    removed = json.loads(mol.GetProp("_poly_csp_removed_old_indices_json"))
    if removed:
        keep_mask = np.ones((coords.shape[0],), dtype=bool)
        keep_mask[np.asarray(removed, dtype=int)] = False
        coords = coords[keep_mask]
    mol = assign_conformer(mol, coords)

    o6_global = residue_atom_global_index(
        residue_index=0,
        monomer_atom_count=template.mol.GetNumAtoms(),
        local_atom_index=template.site_idx["O6"],
    )
    assert mol.GetAtomWithIdx(o6_global).GetTotalNumHs(includeNeighbors=True) == 1

    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    assert mol.GetAtomWithIdx(o6_global).GetTotalNumHs(includeNeighbors=True) == 0
    amide_n = next(
        atom for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_connector_role")
        and atom.GetProp("_poly_csp_connector_role") == "amide_n"
    )
    assert amide_n.GetTotalNumHs(includeNeighbors=True) == 1


@pytest.mark.parametrize("selector_name", ("35dcpc", "3c4mpc"))
def test_attach_selector_supports_additional_csp_catalog_selectors(selector_name: str) -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get(selector_name)

    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    atom_count_before = mol.GetNumAtoms()
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    assert mol.GetNumAtoms() == atom_count_before + selector.mol.GetNumAtoms() - 1
    assert mol.GetNumConformers() == 1
    assert len(Chem.DetectChemistryProblems(mol)) == 0
