from __future__ import annotations

from collections import deque
import json
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.dihedrals import measure_dihedral_rad
from poly_csp.structure.matrix import ScrewTransform


def _selector_instance_local_index_maps(mol: Chem.Mol) -> Dict[int, Dict[int, int]]:
    instances: Dict[int, Dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        instances.setdefault(inst, {})[local] = atom.GetIdx()
    return instances


def _selector_aromatic_rings(
    selector_mol: Chem.Mol,
    *,
    ring_size: int | None = None,
) -> list[tuple[int, ...]]:
    rings: list[tuple[int, ...]] = []
    for ring in selector_mol.GetRingInfo().AtomRings():
        if ring_size is not None and len(ring) != int(ring_size):
            continue
        if not all(selector_mol.GetAtomWithIdx(int(idx)).GetIsAromatic() for idx in ring):
            continue
        rings.append(tuple(int(idx) for idx in ring))
    return rings


def selector_aromatic_ring_planarity(
    mol: Chem.Mol,
    selector_mol: Chem.Mol,
    *,
    ring_size: int = 6,
) -> Dict[str, float | int]:
    if mol.GetNumConformers() == 0:
        return {}

    aromatic_rings = _selector_aromatic_rings(selector_mol, ring_size=ring_size)
    if not aromatic_rings:
        return {}

    instances = _selector_instance_local_index_maps(mol)
    if not instances:
        return {}

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    max_deviations_A: list[float] = []
    rms_deviations_A: list[float] = []

    for mapping in instances.values():
        for ring in aromatic_rings:
            if any(local_idx not in mapping for local_idx in ring):
                continue
            ring_xyz = xyz[np.asarray([mapping[local_idx] for local_idx in ring], dtype=int)]
            centroid = np.mean(ring_xyz, axis=0)
            centered = ring_xyz - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            deviations = np.abs(centered @ normal)
            max_deviations_A.append(float(np.max(deviations)))
            rms_deviations_A.append(float(np.sqrt(np.mean(deviations * deviations))))

    if not max_deviations_A:
        return {}

    max_arr = np.asarray(max_deviations_A, dtype=float)
    rms_arr = np.asarray(rms_deviations_A, dtype=float)
    return {
        "ring_count": int(max_arr.size),
        "template_ring_count": int(len(aromatic_rings)),
        "instance_count": int(len(instances)),
        "max_out_of_plane_A": float(np.max(max_arr)),
        "mean_max_out_of_plane_A": float(np.mean(max_arr)),
        "max_rms_out_of_plane_A": float(np.max(rms_arr)),
        "mean_rms_out_of_plane_A": float(np.mean(rms_arr)),
    }


def selector_aromatic_stacking_metrics(
    mol: Chem.Mol,
    selector_mol: Chem.Mol,
    *,
    threshold_A: float = 4.5,
) -> Dict[str, object]:
    if mol.GetNumConformers() == 0:
        return {}

    aromatic_rings = _selector_aromatic_rings(selector_mol, ring_size=None)
    if not aromatic_rings:
        return {}

    instances = _selector_instance_local_index_maps(mol)
    if len(instances) < 2:
        return {}

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    box_vectors_A = None
    if (
        mol.HasProp("_poly_csp_box_a_A")
        and mol.HasProp("_poly_csp_box_b_A")
        and mol.HasProp("_poly_csp_box_c_A")
    ):
        box_vectors_A = (
            float(mol.GetDoubleProp("_poly_csp_box_a_A")),
            float(mol.GetDoubleProp("_poly_csp_box_b_A")),
            float(mol.GetDoubleProp("_poly_csp_box_c_A")),
        )

    centroids_by_instance: dict[int, list[np.ndarray]] = {}
    for instance_id, mapping in instances.items():
        ring_centroids: list[np.ndarray] = []
        for ring in aromatic_rings:
            if any(local_idx not in mapping for local_idx in ring):
                continue
            ring_xyz = xyz[np.asarray([mapping[local_idx] for local_idx in ring], dtype=int)]
            ring_centroids.append(np.mean(ring_xyz, axis=0))
        if ring_centroids:
            centroids_by_instance[int(instance_id)] = ring_centroids

    if len(centroids_by_instance) < 2:
        return {}

    instance_ids = sorted(centroids_by_instance)
    all_distances_A: list[float] = []
    per_pair_min_A: list[float] = []
    instance_pair_stats: list[dict[str, object]] = []
    threshold = float(threshold_A)

    for left_pos, left_id in enumerate(instance_ids):
        left_centroids = centroids_by_instance[left_id]
        for right_id in instance_ids[left_pos + 1 :]:
            right_centroids = centroids_by_instance[right_id]
            pair_distances_A: list[float] = []
            for left_centroid in left_centroids:
                for right_centroid in right_centroids:
                    delta = minimum_image_delta_A(
                        right_centroid - left_centroid,
                        box_vectors_A,
                    )
                    pair_distances_A.append(float(np.linalg.norm(delta)))
            if not pair_distances_A:
                continue
            pair_min_A = float(min(pair_distances_A))
            pair_below_threshold = sum(
                1 for distance_A in pair_distances_A if distance_A < threshold
            )
            all_distances_A.extend(pair_distances_A)
            per_pair_min_A.append(pair_min_A)
            instance_pair_stats.append(
                {
                    "instance_i": int(left_id),
                    "instance_j": int(right_id),
                    "ring_pair_count": int(len(pair_distances_A)),
                    "ring_pairs_below_threshold": int(pair_below_threshold),
                    "min_centroid_distance_A": pair_min_A,
                }
            )

    if not instance_pair_stats:
        return {}

    return {
        "template_ring_count": int(len(aromatic_rings)),
        "instance_count": int(len(centroids_by_instance)),
        "ring_count": int(sum(len(v) for v in centroids_by_instance.values())),
        "threshold_A": threshold,
        "instance_pair_count": int(len(instance_pair_stats)),
        "ring_pair_count": int(len(all_distances_A)),
        "ring_pairs_below_threshold": int(
            sum(int(item["ring_pairs_below_threshold"]) for item in instance_pair_stats)
        ),
        "instance_pairs_below_threshold": int(
            sum(1 for item in instance_pair_stats if item["ring_pairs_below_threshold"] > 0)
        ),
        "min_centroid_distance_A": float(min(all_distances_A)),
        "mean_centroid_distance_A": float(np.mean(np.asarray(all_distances_A, dtype=float))),
        "mean_min_centroid_distance_A": float(
            np.mean(np.asarray(per_pair_min_A, dtype=float))
        ),
        "instance_pair_stats": instance_pair_stats,
    }


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


def minimum_image_delta_A(
    delta: np.ndarray,
    box_vectors_A: tuple[float, float, float] | None,
) -> np.ndarray:
    out = np.asarray(delta, dtype=float).copy()
    if box_vectors_A is None:
        return out
    for axis, box_length in enumerate(box_vectors_A):
        length = float(box_length)
        if length <= 1e-12:
            continue
        out[..., axis] -= length * np.round(out[..., axis] / length)
    return out


def _wrap_coords_A(
    coords: np.ndarray,
    box_vectors_A: tuple[float, float, float] | None,
) -> np.ndarray:
    out = np.asarray(coords, dtype=float).copy()
    if box_vectors_A is None:
        return out
    for axis, box_length in enumerate(box_vectors_A):
        length = float(box_length)
        if length <= 1e-12:
            continue
        out[:, axis] = np.mod(out[:, axis], length)
    return out


def min_interatomic_distance(
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
    box_vectors_A: tuple[float, float, float] | None = None,
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
        diffs = minimum_image_delta_A(coords[tail] - coords[i], box_vectors_A)
        d2 = np.sum(diffs * diffs, axis=1)
        for k, j in enumerate(tail):
            pair = (int(i), int(j)) if i < j else (int(j), int(i))
            if pair in excluded:
                continue
            dmin = min(dmin, float(np.sqrt(d2[k])))
    return dmin


def _atom_class(atom: Chem.Atom) -> str:
    if atom.HasProp("_poly_csp_manifest_source"):
        return (
            "backbone"
            if atom.GetProp("_poly_csp_manifest_source") == "backbone"
            else "selector"
        )
    return "selector" if atom.HasProp("_poly_csp_selector_instance") else "backbone"


def min_distance_by_class(
    mol: Chem.Mol,
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
    box_vectors_A: tuple[float, float, float] | None = None,
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
            delta = minimum_image_delta_A(
                coords[int(j)] - coords[int(i)],
                box_vectors_A,
            )
            d = float(np.linalg.norm(delta))
            if d < out[key]:
                out[key] = d
    return out


def min_interatomic_distance_fast(
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
    cutoff: float = 2.0,
    box_vectors_A: tuple[float, float, float] | None = None,
) -> float:
    """cKDTree-accelerated minimum distance (sub-cutoff pairs only)."""
    idx = np.where(heavy_mask)[0]
    if idx.size < 2:
        return float("inf")
    excluded = excluded_pairs or set()
    coords_idx = np.asarray(coords[idx], dtype=float)
    if box_vectors_A is not None:
        box = np.asarray(box_vectors_A, dtype=float)
        tree = cKDTree(_wrap_coords_A(coords_idx, box_vectors_A), boxsize=box)
    else:
        tree = cKDTree(coords_idx)
    pairs = tree.query_pairs(r=cutoff)
    if not pairs:
        return cutoff
    dmin = cutoff
    for i_pos, j_pos in pairs:
        i, j = int(idx[i_pos]), int(idx[j_pos])
        pair = (min(i, j), max(i, j))
        if pair in excluded:
            continue
        delta = minimum_image_delta_A(coords[j] - coords[i], box_vectors_A)
        d = float(np.linalg.norm(delta))
        dmin = min(dmin, d)
    return dmin


def min_distance_by_class_fast(
    mol: Chem.Mol,
    coords: np.ndarray,
    heavy_mask: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None = None,
    cutoff: float = 2.0,
    box_vectors_A: tuple[float, float, float] | None = None,
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

    coords_idx = np.asarray(coords[idx], dtype=float)
    if box_vectors_A is not None:
        box = np.asarray(box_vectors_A, dtype=float)
        tree = cKDTree(_wrap_coords_A(coords_idx, box_vectors_A), boxsize=box)
    else:
        tree = cKDTree(coords_idx)
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
        delta = minimum_image_delta_A(coords[j] - coords[i], box_vectors_A)
        d = float(np.linalg.norm(delta))
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


def _resolve_screw_transform(
    mol: Chem.Mol,
    helix: HelixSpec | None,
) -> ScrewTransform:
    if helix is not None:
        return ScrewTransform(theta_rad=helix.theta_rad, rise_A=helix.rise_A)
    if not mol.HasProp("_poly_csp_helix_theta_rad") or not mol.HasProp("_poly_csp_helix_rise_A"):
        raise ValueError(
            "Selector symmetry metric requires helix metadata either via the helix "
            "argument or molecule properties _poly_csp_helix_theta_rad/_poly_csp_helix_rise_A."
        )
    return ScrewTransform(
        theta_rad=float(mol.GetDoubleProp("_poly_csp_helix_theta_rad")),
        rise_A=float(mol.GetDoubleProp("_poly_csp_helix_rise_A")),
    )


def _selector_attachment_maps_by_residue_site(
    mol: Chem.Mol,
) -> dict[tuple[int, str], dict[int, int]]:
    out: dict[tuple[int, str], dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        if not atom.HasProp("_poly_csp_residue_index") or not atom.HasProp("_poly_csp_site"):
            continue
        key = (
            int(atom.GetIntProp("_poly_csp_residue_index")),
            atom.GetProp("_poly_csp_site"),
        )
        out.setdefault(key, {})[int(atom.GetIntProp("_poly_csp_selector_local_idx"))] = int(
            atom.GetIdx()
        )
    return out


def selector_screw_symmetry_rmsd_from_mol(
    mol: Chem.Mol,
    helix: HelixSpec | None = None,
    k: int = 1,
) -> float:
    if mol.GetNumConformers() == 0:
        return 0.0
    if k <= 0:
        return 0.0

    attachment_maps = _selector_attachment_maps_by_residue_site(mol)
    if not attachment_maps:
        return 0.0

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    inv = _resolve_screw_transform(mol, helix)
    inv = ScrewTransform(theta_rad=-inv.theta_rad, rise_A=-inv.rise_A)

    dp = 0
    if mol.HasProp("_poly_csp_dp"):
        dp = int(mol.GetIntProp("_poly_csp_dp"))
    else:
        dp = 1 + max(residue_index for residue_index, _ in attachment_maps)
    if dp <= k:
        return 0.0

    sites = sorted({site for _, site in attachment_maps})
    sum_sq = 0.0
    count = 0
    for residue_index in range(dp - k):
        for site in sites:
            left = attachment_maps.get((residue_index, site))
            right = attachment_maps.get((residue_index + k, site))
            if not left or not right:
                continue
            shared = sorted(set(left).intersection(right))
            if not shared:
                continue
            left_idx = np.asarray([left[local_idx] for local_idx in shared], dtype=int)
            right_idx = np.asarray([right[local_idx] for local_idx in shared], dtype=int)
            left_xyz = xyz[left_idx]
            right_xyz = xyz[right_idx]
            right_mapped = inv.apply(right_xyz, k)
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

    instances = _selector_instance_local_index_maps(mol)

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
