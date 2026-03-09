from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from rdkit import Chem

from poly_csp.ordering.scoring import minimum_image_delta_A
from poly_csp.structure.pbc import get_box_vectors_A
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class HbondMetrics:
    like_satisfied_pairs: int
    geometric_satisfied_pairs: int
    total_pairs: int
    like_fraction: float
    geometric_fraction: float
    mean_like_distance_A: float
    mean_geometric_distance_A: float


def _selector_atom_records(
    mol: Chem.Mol,
    local_indices: Iterable[int],
) -> List[Tuple[int, int]]:
    local_set = set(int(x) for x in local_indices)
    out: List[Tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        local_idx = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        if local_idx in local_set:
            residue = int(atom.GetIntProp("_poly_csp_residue_index"))
            out.append((residue, atom.GetIdx()))
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros((3,), dtype=float)
    return v / n


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    uu = _normalize(u)
    vv = _normalize(v)
    if float(np.linalg.norm(uu)) < 1e-12 or float(np.linalg.norm(vv)) < 1e-12:
        return 0.0
    cosang = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosang)))


def _first_heavy_neighbor_except(
    mol: Chem.Mol,
    atom_idx: int,
    excluded: set[int],
) -> int | None:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    for nbr in atom.GetNeighbors():
        idx = int(nbr.GetIdx())
        if idx in excluded:
            continue
        if nbr.GetAtomicNum() <= 1:
            continue
        return idx
    return None


def _first_attached_hydrogen(mol: Chem.Mol, atom_idx: int) -> int | None:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            return int(nbr.GetIdx())
    return None


def _periodic_residue_gap(
    left: int,
    right: int,
    *,
    dp: int,
    periodic: bool,
) -> int:
    diff = abs(int(left) - int(right))
    if not periodic or dp <= 1:
        return diff
    return min(diff, int(dp) - diff)


def _vector_A(
    xyz: np.ndarray,
    atom_from: int,
    atom_to: int,
    box_vectors_A: tuple[float, float, float] | None,
) -> np.ndarray:
    return minimum_image_delta_A(
        xyz[int(atom_to)] - xyz[int(atom_from)],
        box_vectors_A,
    )


def compute_hbond_metrics(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    max_distance_A: float = 3.3,
    neighbor_window: int = 1,
    min_donor_angle_deg: float = 100.0,
    min_acceptor_angle_deg: float = 90.0,
    box_vectors_A: tuple[float, float, float] | None = None,
    periodic: bool | None = None,
) -> HbondMetrics:
    """
    Pre-organization metrics for selector donor/acceptor pairs:
    - hbond-like: distance threshold only
    - hbond-geometric: distance + donor/acceptor proxy angle thresholds
    """
    if mol.GetNumConformers() == 0:
        return HbondMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    donors = _selector_atom_records(mol, selector.donors)
    acceptors = _selector_atom_records(mol, selector.acceptors)
    if not donors or not acceptors:
        return HbondMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    periodic_mode = bool(
        periodic
        if periodic is not None
        else (
            mol.HasProp("_poly_csp_end_mode")
            and str(mol.GetProp("_poly_csp_end_mode")).strip().lower() == "periodic"
        )
    )
    resolved_box_vectors_A = (
        box_vectors_A
        if box_vectors_A is not None
        else (get_box_vectors_A(mol) if periodic_mode else None)
    )
    dp = int(mol.GetIntProp("_poly_csp_dp")) if mol.HasProp("_poly_csp_dp") else 0
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    total = 0
    satisfied_like = 0
    satisfied_geom = 0
    like_distances: List[float] = []
    geom_distances: List[float] = []

    for d_res, d_idx in donors:
        for a_res, a_idx in acceptors:
            if _periodic_residue_gap(
                d_res,
                a_res,
                dp=dp,
                periodic=periodic_mode,
            ) > int(neighbor_window):
                continue
            if d_idx == a_idx:
                continue
            total += 1
            donor_h = _first_attached_hydrogen(mol, d_idx)
            if donor_h is not None:
                dist = float(
                    np.linalg.norm(
                        _vector_A(
                            xyz,
                            donor_h,
                            a_idx,
                            resolved_box_vectors_A,
                        )
                    )
                )
            else:
                dist = float(
                    np.linalg.norm(
                        _vector_A(
                            xyz,
                            d_idx,
                            a_idx,
                            resolved_box_vectors_A,
                        )
                    )
                )
            if dist > float(max_distance_A):
                continue

            satisfied_like += 1
            like_distances.append(dist)

            a_proxy = _first_heavy_neighbor_except(
                mol=mol,
                atom_idx=a_idx,
                excluded={d_idx},
            )
            if a_proxy is None:
                continue
            if donor_h is not None and a_proxy is not None:
                donor_angle = _angle_deg(
                    _vector_A(xyz, donor_h, d_idx, resolved_box_vectors_A),
                    _vector_A(xyz, donor_h, a_idx, resolved_box_vectors_A),
                )
                acceptor_angle = _angle_deg(
                    _vector_A(xyz, a_idx, donor_h, resolved_box_vectors_A),
                    _vector_A(xyz, a_idx, a_proxy, resolved_box_vectors_A),
                )
            else:
                d_proxy = _first_heavy_neighbor_except(
                    mol=mol,
                    atom_idx=d_idx,
                    excluded={a_idx},
                )
                if d_proxy is None or a_proxy is None:
                    continue
                donor_angle = _angle_deg(
                    _vector_A(xyz, d_proxy, d_idx, resolved_box_vectors_A),
                    _vector_A(xyz, d_idx, a_idx, resolved_box_vectors_A),
                )
                acceptor_angle = _angle_deg(
                    _vector_A(xyz, a_idx, d_idx, resolved_box_vectors_A),
                    _vector_A(xyz, a_idx, a_proxy, resolved_box_vectors_A),
                )
            if (
                donor_angle >= float(min_donor_angle_deg)
                and acceptor_angle >= float(min_acceptor_angle_deg)
            ):
                satisfied_geom += 1
                geom_distances.append(dist)

    like_fraction = float(satisfied_like / total) if total > 0 else 0.0
    geometric_fraction = float(satisfied_geom / total) if total > 0 else 0.0
    return HbondMetrics(
        like_satisfied_pairs=satisfied_like,
        geometric_satisfied_pairs=satisfied_geom,
        total_pairs=total,
        like_fraction=like_fraction,
        geometric_fraction=geometric_fraction,
        mean_like_distance_A=float(np.mean(like_distances)) if like_distances else 0.0,
        mean_geometric_distance_A=float(np.mean(geom_distances))
        if geom_distances
        else 0.0,
    )
