# poly_csp/topology/backbone.py
from __future__ import annotations

import json
from bisect import bisect_left
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.topology.monomers import GlucoseMonomerTemplate


def _new_index(old_idx: int, removed_sorted: list[int]) -> int:
    return old_idx - bisect_left(removed_sorted, old_idx)


def _build_residue_label_map(
    template: GlucoseMonomerTemplate,
    dp: int,
    removed_sorted: list[int],
) -> list[dict[str, int]]:
    n = template.mol.GetNumAtoms()
    removed_set = set(removed_sorted)
    maps: list[dict[str, int]] = []
    for residue_index in range(dp):
        label_map: dict[str, int] = {}
        for label, local_idx in template.atom_idx.items():
            old = residue_index * n + local_idx
            if old in removed_set:
                continue
            label_map[label] = _new_index(old, removed_sorted)
        maps.append(label_map)
    return maps


def _set_polymer_metadata(
    mol: Chem.Mol,
    template: GlucoseMonomerTemplate,
    dp: int,
    removed_sorted: list[int],
) -> None:
    residue_label_map = _build_residue_label_map(template, dp, removed_sorted)

    mol.SetIntProp("_poly_csp_dp", dp)
    mol.SetProp("_poly_csp_polymer", template.polymer)
    mol.SetIntProp("_poly_csp_template_atom_count", template.mol.GetNumAtoms())
    mol.SetProp("_poly_csp_representation", template.representation)
    mol.SetProp("_poly_csp_removed_old_indices_json", json.dumps(removed_sorted))
    mol.SetProp("_poly_csp_residue_label_map_json", json.dumps(residue_label_map))

    # Backward-compatible quick local indices for fixed-site labels.
    for name, idx in template.site_idx.items():
        mol.SetIntProp(f"_poly_csp_siteidx_{name}", int(idx))


def polymerize(
    template: GlucoseMonomerTemplate,
    dp: int,
    linkage: Literal["1-4"] = "1-4",
    anomer: Literal["alpha", "beta"] = "beta",
) -> Chem.Mol:
    """Repeat monomer dp times and add glycosidic bonds deterministically."""
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}")
    if linkage != "1-4":
        raise ValueError(f"Unsupported linkage {linkage!r}; only '1-4' is implemented.")
    if anomer not in {"alpha", "beta"}:
        raise ValueError(f"Unsupported anomer {anomer!r}")

    n_monomer_atoms = template.mol.GetNumAtoms()
    rw = Chem.RWMol()
    for _ in range(dp):
        rw.InsertMol(template.mol)

    # Stage-1 representation uses O4(i)-C1(i+1) linkage.
    for i in range(dp - 1):
        o4_i = i * n_monomer_atoms + template.atom_idx["O4"]
        c1_ip1 = (i + 1) * n_monomer_atoms + template.atom_idx["C1"]
        rw.AddBond(o4_i, c1_ip1, Chem.BondType.SINGLE)

    removed_sorted: list[int] = []
    if template.representation == "natural_oh" and dp > 1:
        # For natural OH templates, remove O1 on residues that receive an incoming
        # glycosidic linkage so C1 valence stays chemically valid.
        if "O1" not in template.atom_idx:
            raise ValueError("natural_oh representation requires O1 in atom_idx.")
        o1_local = template.atom_idx["O1"]
        removed_sorted = sorted(
            [residue_index * n_monomer_atoms + o1_local for residue_index in range(1, dp)]
        )
        for idx in reversed(removed_sorted):
            rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    _set_polymer_metadata(mol, template, dp, removed_sorted)
    return mol


def assign_conformer(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    """Attach coords (N,3) to mol as a single conformer."""
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3); got {xyz.shape}")
    if xyz.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"Atom count mismatch: coords has {xyz.shape[0]}, mol has {mol.GetNumAtoms()}."
        )

    out = Chem.Mol(mol)
    out.RemoveAllConformers()

    conf = Chem.Conformer(out.GetNumAtoms())
    for i in range(out.GetNumAtoms()):
        x, y, z = xyz[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.AddConformer(conf, assignId=True)
    return out
