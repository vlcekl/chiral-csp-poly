from __future__ import annotations

import json

import numpy as np

from poly_csp.config.schema import HelixSpec
from tests.support import build_backbone_coords
from poly_csp.structure.hydrogens import complete_with_hydrogens
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.atom_mapping import attachment_instance_maps
from poly_csp.topology.selectors import SelectorRegistry


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        repeat_residues=4,
        repeat_turns=3,
        axial_repeat_A=14.8,
        handedness="left",
    )


def test_complete_with_hydrogens_adds_backbone_hydroxyl_hydrogens() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    coords = build_backbone_coords(template, _helix(), dp=1)
    mol = assign_conformer(mol, coords)

    all_atom = complete_with_hydrogens(mol, add_coords=True, optimize="h_only")
    maps = json.loads(all_atom.GetProp("_poly_csp_residue_label_map_json"))
    for label in ("O1", "O2", "O3", "O4", "O6"):
        oxygen = all_atom.GetAtomWithIdx(maps[0][label])
        assert sum(1 for nbr in oxygen.GetNeighbors() if nbr.GetAtomicNum() == 1) == 1


def test_complete_with_hydrogens_preserves_heavy_coords_and_metadata() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = SelectorRegistry.get("35dmpc")

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    coords = build_backbone_coords(template, _helix(), dp=2)
    removed = json.loads(mol.GetProp("_poly_csp_removed_old_indices_json"))
    if removed:
        keep_mask = np.ones((coords.shape[0],), dtype=bool)
        keep_mask[np.asarray(removed, dtype=int)] = False
        coords = coords[keep_mask]
    mol = assign_conformer(mol, coords)
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    heavy_xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)
    all_atom = complete_with_hydrogens(mol, add_coords=True, optimize="h_only")

    assert all_atom.GetNumAtoms() > mol.GetNumAtoms()
    all_xyz = np.asarray(all_atom.GetConformer(0).GetPositions(), dtype=float)
    assert np.allclose(all_xyz[: mol.GetNumAtoms()], heavy_xyz, atol=1e-6)

    added_h = [atom for atom in all_atom.GetAtoms() if atom.GetAtomicNum() == 1]
    assert added_h
    assert all(atom.HasProp("_poly_csp_parent_heavy_idx") for atom in added_h)

    amide_n = next(
        atom for atom in all_atom.GetAtoms()
        if atom.HasProp("_poly_csp_connector_role")
        and atom.GetProp("_poly_csp_connector_role") == "amide_n"
    )
    amide_h = [nbr for nbr in amide_n.GetNeighbors() if nbr.GetAtomicNum() == 1]
    assert len(amide_h) == 1
    assert amide_h[0].GetIntProp("_poly_csp_parent_heavy_idx") == amide_n.GetIdx()
    assert amide_h[0].GetProp("_poly_csp_component") == "connector"


def test_complete_with_hydrogens_only_on_atoms_targets_selector_hydrogens() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = SelectorRegistry.get("35dmpc")

    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    coords = build_backbone_coords(template, _helix(), dp=1)
    mol = assign_conformer(mol, coords)
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    selector_atoms = sorted(
        {
            atom_idx
            for mapping in attachment_instance_maps(mol).values()
            for atom_idx in mapping.values()
        }
    )
    targeted = complete_with_hydrogens(
        mol,
        add_coords=True,
        optimize="h_only",
        only_on_atoms=selector_atoms,
    )

    maps = json.loads(targeted.GetProp("_poly_csp_residue_label_map_json"))
    for label in ("O1", "O2", "O3", "O4", "O6"):
        oxygen = targeted.GetAtomWithIdx(maps[0][label])
        assert sum(1 for nbr in oxygen.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0

    amide_n = next(
        atom for atom in targeted.GetAtoms()
        if atom.HasProp("_poly_csp_connector_role")
        and atom.GetProp("_poly_csp_connector_role") == "amide_n"
    )
    assert any(nbr.GetAtomicNum() == 1 for nbr in amide_n.GetNeighbors())
