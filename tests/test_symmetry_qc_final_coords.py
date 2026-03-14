from __future__ import annotations

import json

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from tests.support import build_backbone_coords
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.config.schema import HelixSpec
from poly_csp.ordering.scoring import (
    screw_symmetry_rmsd_from_mol,
    selector_screw_symmetry_rmsd_from_mol,
)
from poly_csp.topology.selectors import SelectorRegistry
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


def _set_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def test_screw_symmetry_rmsd_from_mol_uses_final_coordinates() -> None:
    template = make_glucose_template("amylose")
    helix = _helix()
    dp = 6

    coords = build_backbone_coords(template=template, helix=helix, dp=dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    baseline = screw_symmetry_rmsd_from_mol(mol, helix=helix, k=4)
    assert baseline < 1e-10

    maps = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
    atom_to_perturb = int(maps[4]["C2"])
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    xyz[atom_to_perturb] += np.array([0.35, 0.0, 0.0], dtype=float)
    perturbed = _set_coords(mol, xyz)

    updated = screw_symmetry_rmsd_from_mol(perturbed, helix=helix, k=4)
    assert updated > 1e-3


def test_selector_screw_symmetry_rmsd_from_mol_detects_selector_perturbation() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(
        polymer="amylose",
        dp=3,
        selector=selector,
        site="C6",
    )

    baseline = selector_screw_symmetry_rmsd_from_mol(mol, helix=_helix(), k=1)
    assert baseline > 0.0

    distorted = Chem.Mol(mol)
    conf = distorted.GetConformer()
    atom_idx = next(
        atom.GetIdx()
        for atom in distorted.GetAtoms()
        if atom.HasProp("_poly_csp_selector_instance")
        and int(atom.GetIntProp("_poly_csp_residue_index")) == 1
        and atom.GetProp("_poly_csp_site") == "C6"
    )
    pos = conf.GetAtomPosition(atom_idx)
    conf.SetAtomPosition(atom_idx, Point3D(float(pos.x) + 0.4, float(pos.y), float(pos.z)))

    updated = selector_screw_symmetry_rmsd_from_mol(distorted, helix=_helix(), k=1)
    assert updated > baseline
