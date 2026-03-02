# poly_csp/chemistry/utils.py
"""Shared chemistry utilities used across multiple modules."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from rdkit import Chem


def copy_mol_props(src: Chem.Mol, dst: Chem.Mol) -> None:
    """Copy all private/public properties from *src* to *dst*."""
    props = src.GetPropsAsDict(includePrivate=True, includeComputed=False)
    for key, value in props.items():
        if isinstance(value, bool):
            dst.SetBoolProp(key, bool(value))
        elif isinstance(value, int):
            dst.SetIntProp(key, int(value))
        elif isinstance(value, float):
            dst.SetDoubleProp(key, float(value))
        else:
            dst.SetProp(key, str(value))


def residue_label_maps(mol: Chem.Mol) -> List[Dict[str, int]]:
    """Parse the residue label map metadata from the molecule."""
    if not mol.HasProp("_poly_csp_residue_label_map_json"):
        raise ValueError("Missing _poly_csp_residue_label_map_json metadata on molecule.")
    payload = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
    if not isinstance(payload, list):
        raise ValueError("Invalid residue label map metadata format.")
    maps: List[Dict[str, int]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Invalid residue label map entry.")
        maps.append({str(k): int(v) for k, v in item.items()})
    return maps


def set_residue_label_maps(mol: Chem.Mol, maps: List[Dict[str, int]]) -> None:
    """Write residue label map metadata onto the molecule."""
    mol.SetProp("_poly_csp_residue_label_map_json", json.dumps(maps))


def removed_old_indices(mol: Chem.Mol) -> List[int]:
    """Parse removed-old-indices metadata."""
    if not mol.HasProp("_poly_csp_removed_old_indices_json"):
        return []
    payload = json.loads(mol.GetProp("_poly_csp_removed_old_indices_json"))
    if not isinstance(payload, list):
        raise ValueError("Invalid _poly_csp_removed_old_indices_json metadata.")
    return [int(x) for x in payload]


def set_removed_old_indices(mol: Chem.Mol, removed: List[int]) -> None:
    """Write removed-old-indices metadata."""
    mol.SetProp("_poly_csp_removed_old_indices_json", json.dumps(sorted(set(removed))))


def heavy_atom_mask(mol: Chem.Mol) -> np.ndarray:
    """Boolean mask: True for heavy atoms (Z > 1)."""
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def backbone_heavy_indices(mol: Chem.Mol) -> List[int]:
    """Return global indices of heavy backbone atoms (not selectors)."""
    idx: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.HasProp("_poly_csp_selector_instance"):
            continue
        idx.append(atom.GetIdx())
    return idx


def coords_from_mol(mol: Chem.Mol) -> np.ndarray | None:
    """Extract (N,3) coordinate array from the first conformer, or None."""
    if mol.GetNumConformers() == 0:
        return None
    return np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))


def set_coords(mol: Chem.Mol, coords: np.ndarray | None) -> None:
    """Assign coordinates to the molecule, replacing any existing conformer."""
    from rdkit.Geometry import Point3D

    if coords is None:
        mol.RemoveAllConformers()
        return
    xyz = np.asarray(coords, dtype=float).reshape((-1, 3))
    if xyz.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"Coordinate size mismatch: {xyz.shape[0]} vs {mol.GetNumAtoms()} atoms."
        )
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
