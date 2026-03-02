from __future__ import annotations

from collections import deque
import json
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from poly_csp.config.schema import HelixSpec
from poly_csp.geometry.dihedrals import measure_dihedral_rad
from poly_csp.geometry.transform import ScrewTransform


def bonded_exclusion_pairs(mol: Chem.Mol, max_path_length: int = 2) -> set[tuple[int, int]]:
    """Return atom pairs with shortest bond-path <= max_path_length."""
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    excluded: set[tuple[int, int]] = set()
    for src in range(n):
        q: deque[tuple[int, int]] = deque([(src, 0)])
        seen = {src}
        while q:
            node, depth = q.popleft()
            if depth >= max_path_length:
                continue
            for nbr in adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    q.append((nbr, depth + 1))
                i, j = (src, nbr) if src < nbr else (nbr, src)
                if src != nbr:
                    excluded.add((i, j))
    return excluded


def min_interatomic_distance(
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
) -> float:
    idx = np.where(heavy_mask)[0]
    if idx.size < 2:
        return float("inf")
    excluded = excluded_pairs or set()

    dmin = float("inf")
    for pos_i, i in enumerate(idx):
        tail = idx[pos_i + 1 :]
        if tail.size == 0:
            continue
        diffs = coords[tail] - coords[i]
        d2 = np.sum(diffs * diffs, axis=1)
        for k, j in enumerate(tail):
            pair = (int(i), int(j)) if i < j else (int(j), int(i))
            if pair in excluded:
                continue
            dmin = min(dmin, float(np.sqrt(d2[k])))
    return dmin


def _atom_class(atom: Chem.Atom) -> str:
    return "selector" if atom.HasProp("_poly_csp_selector_instance") else "backbone"


def min_distance_by_class(
    mol: Chem.Mol,
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
) -> Dict[str, float]:
    idx = np.where(heavy_mask)[0]
    excluded = excluded_pairs or set()
    out = {
        "backbone_backbone": float("inf"),
        "backbone_selector": float("inf"),
        "selector_selector": float("inf"),
    }
    for a_pos, i in enumerate(idx):
        ai = mol.GetAtomWithIdx(int(i))
        ci = _atom_class(ai)
        for j in idx[a_pos + 1 :]:
            pair = (int(i), int(j)) if i < j else (int(j), int(i))
            if pair in excluded:
                continue
            aj = mol.GetAtomWithIdx(int(j))
            cj = _atom_class(aj)
            if ci == "backbone" and cj == "backbone":
                key = "backbone_backbone"
            elif ci == "selector" and cj == "selector":
                key = "selector_selector"
            else:
                key = "backbone_selector"
            d = float(np.linalg.norm(coords[int(i)] - coords[int(j)]))
            if d < out[key]:
                out[key] = d
    return out


def min_interatomic_distance_fast(
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
    cutoff: float = 2.0,
) -> float:
    """cKDTree-accelerated minimum distance (sub-cutoff pairs only)."""
    idx = np.where(heavy_mask)[0]
    if idx.size < 2:
        return float("inf")
    excluded = excluded_pairs or set()
    tree = cKDTree(coords[idx])
    pairs = tree.query_pairs(r=cutoff)
    if not pairs:
        return cutoff
    dmin = cutoff
    for i_pos, j_pos in pairs:
        i, j = int(idx[i_pos]), int(idx[j_pos])
        pair = (min(i, j), max(i, j))
        if pair in excluded:
            continue
        d = float(np.linalg.norm(coords[i] - coords[j]))
        dmin = min(dmin, d)
    return dmin


def min_distance_by_class_fast(
    mol: Chem.Mol,
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
    cutoff: float = 2.0,
) -> Dict[str, float]:
    """cKDTree-accelerated class-aware minimum distances."""
    idx = np.where(heavy_mask)[0]
    excluded = excluded_pairs or set()
    out = {
        "backbone_backbone": float("inf"),
        "backbone_selector": float("inf"),
        "selector_selector": float("inf"),
    }
    if idx.size < 2:
        return out

    # pre-compute classes for heavy atoms
    classes = np.array([_atom_class(mol.GetAtomWithIdx(int(i))) for i in idx])

    tree = cKDTree(coords[idx])
    pairs = tree.query_pairs(r=cutoff)
    for i_pos, j_pos in pairs:
        i, j = int(idx[i_pos]), int(idx[j_pos])
        pair = (min(i, j), max(i, j))
        if pair in excluded:
            continue
        ci, cj = classes[i_pos], classes[j_pos]
        if ci == "backbone" and cj == "backbone":
            key = "backbone_backbone"
        elif ci == "selector" and cj == "selector":
            key = "selector_selector"
        else:
            key = "backbone_selector"
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if d < out[key]:
            out[key] = d
    return out


def screw_symmetry_rmsd(
    coords: np.ndarray,
    residue_atom_count: int,
    helix: HelixSpec,
    k: int = 1,
) -> float:
    """
    Compare residue 0 to residue k mapped back by inverse screw, RMSD on atoms.
    """
    if coords.shape[0] < (k + 1) * residue_atom_count:
        return 0.0

    res0 = coords[0:residue_atom_count]
    resk = coords[k * residue_atom_count : (k + 1) * residue_atom_count]

    inv = ScrewTransform(theta_rad=-helix.theta_rad, rise_A=-helix.rise_A)
    resk_mapped = inv.apply(resk, k)

    diff = res0 - resk_mapped
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


_BACKBONE_SYMMETRY_LABELS = (
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "O5",
    "O2",
    "O3",
    "O4",
    "O6",
    "O1",
)


def _residue_label_maps(mol: Chem.Mol) -> list[dict[str, int]]:
    if not mol.HasProp("_poly_csp_residue_label_map_json"):
        raise ValueError("Missing _poly_csp_residue_label_map_json metadata on molecule.")
    payload = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
    if not isinstance(payload, list):
        raise ValueError("Invalid residue label map metadata format.")
    maps: list[dict[str, int]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Invalid residue label map entry.")
        maps.append({str(k): int(v) for k, v in item.items()})
    return maps


def screw_symmetry_rmsd_from_mol(
    mol: Chem.Mol,
    helix: HelixSpec,
    k: int | None = None,
) -> float:
    """
    Evaluate screw symmetry on the final molecule coordinates.
    Uses residue label maps and compares only shared backbone labels.
    """
    if mol.GetNumConformers() == 0:
        return 0.0

    maps = _residue_label_maps(mol)
    dp = len(maps)
    step = int(k if k is not None else (helix.repeat_residues or 1))
    if step <= 0 or dp <= step:
        return 0.0

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    inv = ScrewTransform(theta_rad=-helix.theta_rad, rise_A=-helix.rise_A)

    sum_sq = 0.0
    count = 0
    for i in range(dp - step):
        left = maps[i]
        right = maps[i + step]
        shared = [label for label in _BACKBONE_SYMMETRY_LABELS if label in left and label in right]
        if not shared:
            continue
        left_idx = np.asarray([left[label] for label in shared], dtype=int)
        right_idx = np.asarray([right[label] for label in shared], dtype=int)

        left_xyz = xyz[left_idx]
        right_xyz = xyz[right_idx]
        right_mapped = inv.apply(right_xyz, step)

        diff = left_xyz - right_mapped
        sum_sq += float(np.sum(diff * diff))
        count += int(diff.shape[0])

    if count == 0:
        return 0.0
    return float(np.sqrt(sum_sq / float(count)))


def selector_torsion_stats(
    mol: Chem.Mol,
    selector_dihedrals: Dict[str, tuple[int, int, int, int]],
    attach_dummy_idx: int | None,
) -> Dict[str, Dict[str, float]]:
    if mol.GetNumConformers() == 0:
        return {}
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))

    instances: Dict[int, Dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        instances.setdefault(inst, {})[local] = atom.GetIdx()

    values_deg: Dict[str, List[float]] = {name: [] for name in selector_dihedrals}
    for mapping in instances.values():
        for name, (a_l, b_l, c_l, d_l) in selector_dihedrals.items():
            local = (a_l, b_l, c_l, d_l)
            if attach_dummy_idx is not None and attach_dummy_idx in local:
                # Skip dummy-dependent torsions in aggregate statistics.
                continue
            if any(idx not in mapping for idx in local):
                continue
            a, b, c, d = (mapping[a_l], mapping[b_l], mapping[c_l], mapping[d_l])
            angle = np.rad2deg(measure_dihedral_rad(xyz, a, b, c, d))
            values_deg[name].append(float(angle))

    out: Dict[str, Dict[str, float]] = {}
    for name, vals in values_deg.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[name] = {
            "count": float(arr.size),
            "mean_deg": float(arr.mean()),
            "std_deg": float(arr.std()),
            "min_deg": float(arr.min()),
            "max_deg": float(arr.max()),
        }
    return out
