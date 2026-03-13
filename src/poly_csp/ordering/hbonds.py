from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from rdkit import Chem

from poly_csp.config.schema import HbondPairingMode, HbondRestraintAtomMode
from poly_csp.ordering.scoring import minimum_image_delta_A
from poly_csp.structure.pbc import get_box_vectors_A
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class HbondMetrics:
    like_satisfied_pairs: int
    geometric_satisfied_pairs: int
    total_pairs: int
    donor_count: int
    like_satisfied_donors: int
    geometric_satisfied_donors: int
    like_fraction: float
    geometric_fraction: float
    like_donor_occupancy_fraction: float
    geometric_donor_occupancy_fraction: float
    mean_like_distance_A: float
    mean_geometric_distance_A: float


@dataclass(frozen=True)
class HbondAtomRecord:
    residue_index: int
    instance_id: int
    atom_idx: int
    hydrogen_idx: int | None = None


_DONOR_HEAVY_IDEAL_OFFSET_NM = 0.10


def _selector_atom_records(
    mol: Chem.Mol,
    local_indices: Iterable[int],
    *,
    include_attached_hydrogen: bool = False,
) -> List[HbondAtomRecord]:
    local_set = set(int(x) for x in local_indices)
    out: List[HbondAtomRecord] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        local_idx = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        if local_idx in local_set:
            out.append(
                HbondAtomRecord(
                    residue_index=int(atom.GetIntProp("_poly_csp_residue_index")),
                    instance_id=int(atom.GetIntProp("_poly_csp_selector_instance")),
                    atom_idx=int(atom.GetIdx()),
                    hydrogen_idx=(
                        _first_attached_hydrogen(mol, int(atom.GetIdx()))
                        if include_attached_hydrogen
                        else None
                    ),
                )
            )
    return out


def selector_hbond_atom_records(
    mol: Chem.Mol,
    selector: SelectorTemplate,
) -> tuple[list[HbondAtomRecord], list[HbondAtomRecord]]:
    return (
        _selector_atom_records(
            mol,
            selector.donors,
            include_attached_hydrogen=True,
        ),
        _selector_atom_records(
            mol,
            selector.acceptors,
            include_attached_hydrogen=False,
        ),
    )


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


def _resolved_hbond_context(
    mol: Chem.Mol,
    *,
    box_vectors_A: tuple[float, float, float] | None,
    periodic: bool | None,
) -> tuple[bool, tuple[float, float, float] | None, int]:
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
    return periodic_mode, resolved_box_vectors_A, dp


def _restraint_donor_atom_idx(
    donor: HbondAtomRecord,
    atom_mode: HbondRestraintAtomMode,
) -> int:
    if atom_mode == "hydrogen_if_present" and donor.hydrogen_idx is not None:
        return int(donor.hydrogen_idx)
    return int(donor.atom_idx)


def _resolved_restraint_target_nm(
    *,
    measured_distance_nm: float,
    ideal_target_nm: float | None,
    donor: HbondAtomRecord,
    atom_mode: HbondRestraintAtomMode,
) -> float:
    if ideal_target_nm is None:
        return float(measured_distance_nm)

    target_nm = float(ideal_target_nm)
    if atom_mode == "hydrogen_if_present" and donor.hydrogen_idx is not None:
        return target_nm

    # Interpret short ideal targets as H...O distances and convert them to a
    # donor-heavy-atom restraint target when the hydrogen cannot be used.
    if target_nm < 0.24:
        return target_nm + _DONOR_HEAVY_IDEAL_OFFSET_NM
    return target_nm


def build_hbond_restraint_pairs(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    *,
    max_distance_A: float = 3.3,
    neighbor_window: int = 1,
    pairing_mode: HbondPairingMode = "legacy_all_pairs",
    atom_mode: HbondRestraintAtomMode = "hydrogen_if_present",
    ideal_target_nm: float | None = None,
    box_vectors_A: tuple[float, float, float] | None = None,
    periodic: bool | None = None,
) -> list[tuple[int, int, float]]:
    if mol.GetNumConformers() == 0:
        return []

    donors, acceptors = selector_hbond_atom_records(mol, selector)
    if not donors or not acceptors:
        return []

    periodic_mode, resolved_box_vectors_A, dp = _resolved_hbond_context(
        mol,
        box_vectors_A=box_vectors_A,
        periodic=periodic,
    )
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    candidates: list[
        tuple[float, int, int, int, int, int, int, float]
    ] = []

    for donor in donors:
        donor_atom_idx = _restraint_donor_atom_idx(donor, atom_mode)
        for acceptor in acceptors:
            if donor.instance_id == acceptor.instance_id:
                continue
            if _periodic_residue_gap(
                donor.residue_index,
                acceptor.residue_index,
                dp=dp,
                periodic=periodic_mode,
            ) > int(neighbor_window):
                continue
            if donor_atom_idx == acceptor.atom_idx:
                continue
            distance_A = float(
                np.linalg.norm(
                    _vector_A(
                        xyz,
                        donor_atom_idx,
                        acceptor.atom_idx,
                        resolved_box_vectors_A,
                    )
                )
            )
            if distance_A > float(max_distance_A):
                continue
            candidates.append(
                (
                    distance_A,
                    donor.residue_index,
                    donor.atom_idx,
                    acceptor.residue_index,
                    acceptor.atom_idx,
                    donor.atom_idx,
                    donor_atom_idx,
                    _resolved_restraint_target_nm(
                        measured_distance_nm=(distance_A / 10.0),
                        ideal_target_nm=ideal_target_nm,
                        donor=donor,
                        atom_mode=atom_mode,
                    ),
                )
            )

    if not candidates:
        return []

    if pairing_mode == "nearest_unique":
        selected: list[tuple[int, int, float]] = []
        used_donors: set[int] = set()
        used_acceptors: set[int] = set()
        for _, _, _, _, acceptor_atom_idx, donor_key, donor_atom_idx, target_nm in sorted(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
            ),
        ):
            if donor_key in used_donors or acceptor_atom_idx in used_acceptors:
                continue
            used_donors.add(donor_key)
            used_acceptors.add(acceptor_atom_idx)
            selected.append((int(donor_atom_idx), int(acceptor_atom_idx), float(target_nm)))
        return selected

    return [
        (int(donor_atom_idx), int(acceptor_atom_idx), float(target_nm))
        for _, _, _, _, acceptor_atom_idx, _, donor_atom_idx, target_nm in sorted(
            candidates,
            key=lambda item: (
                item[1],
                item[2],
                item[3],
                item[4],
            ),
        )
    ]


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
        return HbondMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    donors, acceptors = selector_hbond_atom_records(mol, selector)
    if not donors or not acceptors:
        return HbondMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    periodic_mode, resolved_box_vectors_A, dp = _resolved_hbond_context(
        mol,
        box_vectors_A=box_vectors_A,
        periodic=periodic,
    )
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    total = 0
    satisfied_like = 0
    satisfied_geom = 0
    donor_records = {
        (int(donor.residue_index), int(donor.atom_idx))
        for donor in donors
    }
    like_satisfied_donors: set[tuple[int, int]] = set()
    geometric_satisfied_donors: set[tuple[int, int]] = set()
    like_distances: List[float] = []
    geom_distances: List[float] = []

    for donor in donors:
        for acceptor in acceptors:
            if _periodic_residue_gap(
                donor.residue_index,
                acceptor.residue_index,
                dp=dp,
                periodic=periodic_mode,
            ) > int(neighbor_window):
                continue
            if donor.atom_idx == acceptor.atom_idx:
                continue
            total += 1
            donor_h = donor.hydrogen_idx
            if donor_h is not None:
                dist = float(
                    np.linalg.norm(
                        _vector_A(
                            xyz,
                            donor_h,
                            acceptor.atom_idx,
                            resolved_box_vectors_A,
                        )
                    )
                )
            else:
                dist = float(
                    np.linalg.norm(
                        _vector_A(
                            xyz,
                            donor.atom_idx,
                            acceptor.atom_idx,
                            resolved_box_vectors_A,
                        )
                    )
                )
            if dist > float(max_distance_A):
                continue

            satisfied_like += 1
            like_satisfied_donors.add((int(donor.residue_index), int(donor.atom_idx)))
            like_distances.append(dist)

            a_proxy = _first_heavy_neighbor_except(
                mol=mol,
                atom_idx=acceptor.atom_idx,
                excluded={donor.atom_idx},
            )
            if a_proxy is None:
                continue
            if donor_h is not None:
                donor_angle = _angle_deg(
                    _vector_A(
                        xyz,
                        donor_h,
                        donor.atom_idx,
                        resolved_box_vectors_A,
                    ),
                    _vector_A(
                        xyz,
                        donor_h,
                        acceptor.atom_idx,
                        resolved_box_vectors_A,
                    ),
                )
                acceptor_angle = _angle_deg(
                    _vector_A(
                        xyz,
                        acceptor.atom_idx,
                        donor_h,
                        resolved_box_vectors_A,
                    ),
                    _vector_A(
                        xyz,
                        acceptor.atom_idx,
                        a_proxy,
                        resolved_box_vectors_A,
                    ),
                )
            else:
                d_proxy = _first_heavy_neighbor_except(
                    mol=mol,
                    atom_idx=donor.atom_idx,
                    excluded={acceptor.atom_idx},
                )
                if d_proxy is None or a_proxy is None:
                    continue
                donor_angle = _angle_deg(
                    _vector_A(
                        xyz,
                        d_proxy,
                        donor.atom_idx,
                        resolved_box_vectors_A,
                    ),
                    _vector_A(
                        xyz,
                        donor.atom_idx,
                        acceptor.atom_idx,
                        resolved_box_vectors_A,
                    ),
                )
                acceptor_angle = _angle_deg(
                    _vector_A(
                        xyz,
                        acceptor.atom_idx,
                        donor.atom_idx,
                        resolved_box_vectors_A,
                    ),
                    _vector_A(
                        xyz,
                        acceptor.atom_idx,
                        a_proxy,
                        resolved_box_vectors_A,
                    ),
                )
            if (
                donor_angle >= float(min_donor_angle_deg)
                and acceptor_angle >= float(min_acceptor_angle_deg)
            ):
                satisfied_geom += 1
                geometric_satisfied_donors.add(
                    (int(donor.residue_index), int(donor.atom_idx))
                )
                geom_distances.append(dist)

    like_fraction = float(satisfied_like / total) if total > 0 else 0.0
    geometric_fraction = float(satisfied_geom / total) if total > 0 else 0.0
    donor_count = len(donor_records)
    like_donor_occupancy = (
        float(len(like_satisfied_donors) / donor_count) if donor_count > 0 else 0.0
    )
    geometric_donor_occupancy = (
        float(len(geometric_satisfied_donors) / donor_count) if donor_count > 0 else 0.0
    )
    return HbondMetrics(
        like_satisfied_pairs=satisfied_like,
        geometric_satisfied_pairs=satisfied_geom,
        total_pairs=total,
        donor_count=donor_count,
        like_satisfied_donors=len(like_satisfied_donors),
        geometric_satisfied_donors=len(geometric_satisfied_donors),
        like_fraction=like_fraction,
        geometric_fraction=geometric_fraction,
        like_donor_occupancy_fraction=like_donor_occupancy,
        geometric_donor_occupancy_fraction=geometric_donor_occupancy,
        mean_like_distance_A=float(np.mean(like_distances)) if like_distances else 0.0,
        mean_geometric_distance_A=float(np.mean(geom_distances))
        if geom_distances
        else 0.0,
    )
