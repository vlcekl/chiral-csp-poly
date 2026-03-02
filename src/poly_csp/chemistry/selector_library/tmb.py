# poly_csp/chemistry/selector_library/tmb.py
"""Tris(4-methylbenzoate) ester selector template.

This validates the selector plugin interface with an ester linkage
(vs. carbamate for 3,5-DMPC).

SMILES: [*:1][C:2](=[O:3])c1ccc(C)cc1
         dummy  C_carbonyl  =O   aromatic ring with para-methyl
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from poly_csp.chemistry.selectors import SelectorTemplate


_TMB_MAPPED_SMILES = "[*:1][C:2](=[O:3])[c:4]1[cH:5][c:6]([CH3:7])[cH:8][cH:9][cH:10]1"


def _idx_from_mapnum(mol: Chem.Mol, map_num: int) -> int:
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise ValueError(f"Map number {map_num} not found in TMB template.")


def _embed_if_needed(mol: Chem.Mol) -> Chem.Mol:
    if mol.GetNumConformers() > 0:
        return mol

    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 4201
    status = AllChem.EmbedMolecule(with_h, params)
    if status != 0:
        status = AllChem.EmbedMolecule(with_h, useRandomCoords=True, randomSeed=4201)
    if status != 0:
        raise RuntimeError("RDKit failed to embed TMB selector template.")
    if all(atom.GetAtomicNum() > 0 for atom in with_h.GetAtoms()):
        AllChem.UFFOptimizeMolecule(with_h, maxIters=250)
    return Chem.RemoveHs(with_h, sanitize=True)


def make_tmb_template() -> SelectorTemplate:
    """Create a tris(4-methylbenzoate) ester selector template."""
    mol = Chem.MolFromSmiles(_TMB_MAPPED_SMILES)
    if mol is None:
        raise ValueError("Could not parse TMB mapped SMILES.")
    Chem.SanitizeMol(mol)
    mol = _embed_if_needed(mol)

    dummy = _idx_from_mapnum(mol, 1)
    carbonyl_c = _idx_from_mapnum(mol, 2)
    carbonyl_o = _idx_from_mapnum(mol, 3)
    c_ipso = _idx_from_mapnum(mol, 4)
    c_ortho = _idx_from_mapnum(mol, 5)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    dihedrals = {
        "tau_ar": (dummy, carbonyl_c, c_ipso, c_ortho),
    }

    return SelectorTemplate(
        name="tmb",
        mol=mol,
        attach_atom_idx=carbonyl_c,
        attach_dummy_idx=dummy,
        dihedrals=dihedrals,
        donors=(),            # ester has no H-bond donors
        acceptors=(carbonyl_o,),
    )
