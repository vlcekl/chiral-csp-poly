from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

import numpy as np
from rdkit import Chem

from poly_csp.config.schema import HbondPairingMode, HbondRestraintAtomMode
from poly_csp.ordering.scoring import minimum_image_delta_A
from poly_csp.structure.matrix import ScrewTransform
from poly_csp.structure.pbc import get_box_vectors_A
from poly_csp.topology.selectors import SelectorTemplate


HbondConnectivityPolicy = Literal["auto", "generic", "csp_literature_v1"]


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
class SelectorHbondDiagnostics:
    metrics: HbondMetrics
    applied_policy: Literal["generic", "csp_literature_v1"]
    family_metrics: dict[str, HbondMetrics]


@dataclass(frozen=True)
class HbondAtomRecord:
    residue_index: int
    instance_id: int
    atom_idx: int
    hydrogen_idx: int | None = None


@dataclass(frozen=True)
class _TargetHbondEdge:
    donor_residue_index: int
    donor_site: str
    donor_atom_idx: int
    donor_hydrogen_idx: int | None
    acceptor_residue_index: int
    acceptor_site: str
    acceptor_atom_idx: int
    cell_shift_steps: int


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


def _connector_role_records(
    mol: Chem.Mol,
    role: str,
    *,
    include_attached_hydrogen: bool = False,
) -> dict[tuple[int, str], HbondAtomRecord]:
    out: dict[tuple[int, str], HbondAtomRecord] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if not atom.HasProp("_poly_csp_connector_role"):
            continue
        if atom.GetProp("_poly_csp_connector_role") != str(role):
            continue
        if not atom.HasProp("_poly_csp_residue_index") or not atom.HasProp("_poly_csp_site"):
            continue
        residue_index = int(atom.GetIntProp("_poly_csp_residue_index"))
        site = str(atom.GetProp("_poly_csp_site"))
        out[(residue_index, site)] = HbondAtomRecord(
            residue_index=residue_index,
            instance_id=(
                int(atom.GetIntProp("_poly_csp_selector_instance"))
                if atom.HasProp("_poly_csp_selector_instance")
                else -1
            ),
            atom_idx=int(atom.GetIdx()),
            hydrogen_idx=(
                _first_attached_hydrogen(mol, int(atom.GetIdx()))
                if include_attached_hydrogen
                else None
            ),
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


def _active_connector_sites(
    mol: Chem.Mol,
    *,
    requested_sites: Iterable[str] | None = None,
) -> set[str]:
    available = {
        str(atom.GetProp("_poly_csp_site"))
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_connector_role") and atom.HasProp("_poly_csp_site")
    }
    if requested_sites is None:
        return available
    requested = {str(site) for site in requested_sites}
    return available & requested


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


def _wrapped_target_residue(
    target_residue_index: int,
    *,
    dp: int,
    periodic: bool,
) -> tuple[int, int] | None:
    if dp <= 0:
        return None
    if not periodic:
        if target_residue_index < 0 or target_residue_index >= dp:
            return None
        return int(target_residue_index), 0
    wrap_count, wrapped = divmod(int(target_residue_index), int(dp))
    return int(wrapped), int(wrap_count) * int(dp)


def _resolved_selector_screw(mol: Chem.Mol) -> ScrewTransform | None:
    if not mol.HasProp("_poly_csp_helix_theta_rad") or not mol.HasProp("_poly_csp_helix_rise_A"):
        return None
    return ScrewTransform(
        theta_rad=float(mol.GetDoubleProp("_poly_csp_helix_theta_rad")),
        rise_A=float(mol.GetDoubleProp("_poly_csp_helix_rise_A")),
    )


def _shifted_point_A(
    xyz: np.ndarray,
    atom_idx: int,
    *,
    screw: ScrewTransform | None,
    cell_shift_steps: int,
) -> np.ndarray:
    point = np.asarray(xyz[int(atom_idx)], dtype=float).reshape((1, 3))
    if int(cell_shift_steps) == 0:
        return point[0]
    if screw is None:
        raise ValueError(
            "Connectivity-aware H-bond metrics require helix screw metadata on the molecule."
        )
    return np.asarray(screw.apply(point, int(cell_shift_steps)), dtype=float)[0]


def _csp_pitch_repeat_residues(mol: Chem.Mol) -> int | None:
    if mol.HasProp("_poly_csp_helix_repeat_residues"):
        repeat = int(mol.GetIntProp("_poly_csp_helix_repeat_residues"))
        return repeat if repeat > 0 else None
    if not mol.HasProp("_poly_csp_polymer"):
        return None
    polymer = str(mol.GetProp("_poly_csp_polymer")).strip().lower()
    if polymer == "amylose":
        return 4
    if polymer == "cellulose":
        return 3
    return None


def _build_csp_target_hbond_edges(
    mol: Chem.Mol,
    *,
    requested_sites: Iterable[str] | None = None,
) -> tuple[_TargetHbondEdge, ...]:
    if not mol.HasProp("_poly_csp_polymer"):
        return ()
    polymer = str(mol.GetProp("_poly_csp_polymer")).strip().lower()
    if polymer not in {"amylose", "cellulose"}:
        return ()
    if mol.GetNumAtoms() == 0:
        return ()

    periodic_mode, _, dp = _resolved_hbond_context(
        mol,
        box_vectors_A=None,
        periodic=None,
    )
    if dp <= 0:
        return ()

    donors = _connector_role_records(
        mol,
        "amide_n",
        include_attached_hydrogen=True,
    )
    acceptors = _connector_role_records(
        mol,
        "carbonyl_o",
        include_attached_hydrogen=False,
    )
    active_sites = _active_connector_sites(mol, requested_sites=requested_sites)
    if not donors or not acceptors or not active_sites:
        return ()

    edges: list[_TargetHbondEdge] = []

    def _append_edge(
        *,
        donor_site: str,
        donor_residue_index: int,
        acceptor_site: str,
        acceptor_residue_index: int,
    ) -> None:
        donor = donors.get((int(donor_residue_index), str(donor_site)))
        if donor is None:
            return
        resolved = _wrapped_target_residue(
            int(acceptor_residue_index),
            dp=dp,
            periodic=periodic_mode,
        )
        if resolved is None:
            return
        acceptor_residue_wrapped, cell_shift_steps = resolved
        acceptor = acceptors.get((int(acceptor_residue_wrapped), str(acceptor_site)))
        if acceptor is None:
            return
        edges.append(
            _TargetHbondEdge(
                donor_residue_index=int(donor_residue_index),
                donor_site=str(donor_site),
                donor_atom_idx=int(donor.atom_idx),
                donor_hydrogen_idx=(
                    None if donor.hydrogen_idx is None else int(donor.hydrogen_idx)
                ),
                acceptor_residue_index=int(acceptor_residue_wrapped),
                acceptor_site=str(acceptor_site),
                acceptor_atom_idx=int(acceptor.atom_idx),
                cell_shift_steps=int(cell_shift_steps),
            )
        )

    if "C2" in active_sites and "C3" in active_sites:
        for residue_index in range(int(dp)):
            _append_edge(
                donor_site="C2",
                donor_residue_index=residue_index,
                acceptor_site="C3",
                acceptor_residue_index=(residue_index - 1),
            )
            _append_edge(
                donor_site="C3",
                donor_residue_index=residue_index,
                acceptor_site="C2",
                acceptor_residue_index=(residue_index - 1),
            )

    pitch_repeat = _csp_pitch_repeat_residues(mol)
    if "C6" in active_sites and pitch_repeat is not None and pitch_repeat > 0:
        for residue_index in range(int(dp)):
            _append_edge(
                donor_site="C6",
                donor_residue_index=residue_index,
                acceptor_site="C6",
                acceptor_residue_index=(residue_index + int(pitch_repeat)),
            )

    return tuple(edges)


def _target_hbond_edge_family(edge: _TargetHbondEdge) -> str:
    donor_site = str(edge.donor_site)
    acceptor_site = str(edge.acceptor_site)
    if donor_site == "C2" and acceptor_site == "C3":
        return "c2_to_c3_zipper"
    if donor_site == "C3" and acceptor_site == "C2":
        return "c3_to_c2_zipper"
    if donor_site == "C6" and acceptor_site == "C6":
        return "c6_pitch_bridge"
    return f"{donor_site.lower()}_to_{acceptor_site.lower()}"


def _zero_hbond_metrics() -> HbondMetrics:
    return HbondMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _hbond_metrics_from_counts(
    *,
    total: int,
    donor_records: set[tuple[int, int]],
    satisfied_like: int,
    satisfied_geom: int,
    like_satisfied_donors: set[tuple[int, int]],
    geometric_satisfied_donors: set[tuple[int, int]],
    like_distances: list[float],
    geom_distances: list[float],
) -> HbondMetrics:
    donor_count = len(donor_records)
    like_fraction = float(satisfied_like / total) if total > 0 else 0.0
    geometric_fraction = float(satisfied_geom / total) if total > 0 else 0.0
    like_donor_occupancy = (
        float(len(like_satisfied_donors) / donor_count) if donor_count > 0 else 0.0
    )
    geometric_donor_occupancy = (
        float(len(geometric_satisfied_donors) / donor_count) if donor_count > 0 else 0.0
    )
    return HbondMetrics(
        like_satisfied_pairs=int(satisfied_like),
        geometric_satisfied_pairs=int(satisfied_geom),
        total_pairs=int(total),
        donor_count=int(donor_count),
        like_satisfied_donors=int(len(like_satisfied_donors)),
        geometric_satisfied_donors=int(len(geometric_satisfied_donors)),
        like_fraction=float(like_fraction),
        geometric_fraction=float(geometric_fraction),
        like_donor_occupancy_fraction=float(like_donor_occupancy),
        geometric_donor_occupancy_fraction=float(geometric_donor_occupancy),
        mean_like_distance_A=float(np.mean(like_distances)) if like_distances else 0.0,
        mean_geometric_distance_A=float(np.mean(geom_distances)) if geom_distances else 0.0,
    )


def _compute_target_hbond_metrics(
    mol: Chem.Mol,
    *,
    target_edges: Iterable[_TargetHbondEdge],
    max_distance_A: float,
    min_donor_angle_deg: float,
    min_acceptor_angle_deg: float,
) -> HbondMetrics:
    if mol.GetNumConformers() == 0:
        return _zero_hbond_metrics(), {}

    edges = tuple(target_edges)
    if not edges:
        return _zero_hbond_metrics(), {}

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    screw = (
        _resolved_selector_screw(mol)
        if any(int(edge.cell_shift_steps) != 0 for edge in edges)
        else None
    )

    total = len(edges)
    satisfied_like = 0
    satisfied_geom = 0
    donor_records = {
        (int(edge.donor_residue_index), int(edge.donor_atom_idx))
        for edge in edges
    }
    like_satisfied_donors: set[tuple[int, int]] = set()
    geometric_satisfied_donors: set[tuple[int, int]] = set()
    like_distances: list[float] = []
    geom_distances: list[float] = []
    family_state: dict[str, dict[str, object]] = {}

    for edge in edges:
        family_name = _target_hbond_edge_family(edge)
        state = family_state.setdefault(
            family_name,
            {
                "total": 0,
                "donor_records": set(),
                "satisfied_like": 0,
                "satisfied_geom": 0,
                "like_satisfied_donors": set(),
                "geometric_satisfied_donors": set(),
                "like_distances": [],
                "geom_distances": [],
            },
        )
        state["total"] = int(state["total"]) + 1
        state["donor_records"].add((int(edge.donor_residue_index), int(edge.donor_atom_idx)))
        donor_anchor_idx = (
            int(edge.donor_hydrogen_idx)
            if edge.donor_hydrogen_idx is not None
            else int(edge.donor_atom_idx)
        )
        donor_pos = np.asarray(xyz[donor_anchor_idx], dtype=float)
        donor_heavy_pos = np.asarray(xyz[int(edge.donor_atom_idx)], dtype=float)
        acceptor_pos = _shifted_point_A(
            xyz,
            int(edge.acceptor_atom_idx),
            screw=screw,
            cell_shift_steps=int(edge.cell_shift_steps),
        )
        dist = float(np.linalg.norm(acceptor_pos - donor_pos))
        if dist > float(max_distance_A):
            continue

        satisfied_like += 1
        like_satisfied_donors.add((int(edge.donor_residue_index), int(edge.donor_atom_idx)))
        like_distances.append(dist)
        state["satisfied_like"] = int(state["satisfied_like"]) + 1
        state["like_satisfied_donors"].add(
            (int(edge.donor_residue_index), int(edge.donor_atom_idx))
        )
        state["like_distances"].append(dist)

        a_proxy = _first_heavy_neighbor_except(
            mol=mol,
            atom_idx=int(edge.acceptor_atom_idx),
            excluded=set(),
        )
        if a_proxy is None:
            continue
        a_proxy_pos = _shifted_point_A(
            xyz,
            int(a_proxy),
            screw=screw,
            cell_shift_steps=int(edge.cell_shift_steps),
        )

        if edge.donor_hydrogen_idx is not None:
            donor_angle = _angle_deg(
                donor_heavy_pos - donor_pos,
                acceptor_pos - donor_pos,
            )
            acceptor_angle = _angle_deg(
                donor_pos - acceptor_pos,
                a_proxy_pos - acceptor_pos,
            )
        else:
            d_proxy = _first_heavy_neighbor_except(
                mol=mol,
                atom_idx=int(edge.donor_atom_idx),
                excluded=set(),
            )
            if d_proxy is None:
                continue
            donor_proxy_pos = np.asarray(xyz[int(d_proxy)], dtype=float)
            donor_angle = _angle_deg(
                donor_proxy_pos - donor_heavy_pos,
                acceptor_pos - donor_heavy_pos,
            )
            acceptor_angle = _angle_deg(
                donor_heavy_pos - acceptor_pos,
                a_proxy_pos - acceptor_pos,
            )

        if (
            donor_angle >= float(min_donor_angle_deg)
            and acceptor_angle >= float(min_acceptor_angle_deg)
        ):
            satisfied_geom += 1
            geometric_satisfied_donors.add(
                (int(edge.donor_residue_index), int(edge.donor_atom_idx))
            )
            geom_distances.append(dist)
            state["satisfied_geom"] = int(state["satisfied_geom"]) + 1
            state["geometric_satisfied_donors"].add(
                (int(edge.donor_residue_index), int(edge.donor_atom_idx))
            )
            state["geom_distances"].append(dist)

    family_metrics = {
        family_name: _hbond_metrics_from_counts(
            total=int(state["total"]),
            donor_records=set(state["donor_records"]),
            satisfied_like=int(state["satisfied_like"]),
            satisfied_geom=int(state["satisfied_geom"]),
            like_satisfied_donors=set(state["like_satisfied_donors"]),
            geometric_satisfied_donors=set(state["geometric_satisfied_donors"]),
            like_distances=list(state["like_distances"]),
            geom_distances=list(state["geom_distances"]),
        )
        for family_name, state in family_state.items()
    }
    return (
        _hbond_metrics_from_counts(
            total=int(total),
            donor_records=set(donor_records),
            satisfied_like=int(satisfied_like),
            satisfied_geom=int(satisfied_geom),
            like_satisfied_donors=set(like_satisfied_donors),
            geometric_satisfied_donors=set(geometric_satisfied_donors),
            like_distances=list(like_distances),
            geom_distances=list(geom_distances),
        ),
        family_metrics,
    )


def resolve_hbond_connectivity_policy(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    *,
    requested_policy: HbondConnectivityPolicy = "auto",
    requested_sites: Iterable[str] | None = None,
) -> Literal["generic", "csp_literature_v1"]:
    if str(requested_policy) == "generic":
        return "generic"
    if selector.linkage_type != "carbamate":
        return "generic"
    target_edges = _build_csp_target_hbond_edges(
        mol,
        requested_sites=requested_sites,
    )
    if target_edges:
        return "csp_literature_v1"
    return "generic"


def compute_selector_hbond_metrics(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    *,
    max_distance_A: float = 3.3,
    neighbor_window: int = 1,
    min_donor_angle_deg: float = 100.0,
    min_acceptor_angle_deg: float = 90.0,
    box_vectors_A: tuple[float, float, float] | None = None,
    periodic: bool | None = None,
    connectivity_policy: HbondConnectivityPolicy = "generic",
    requested_sites: Iterable[str] | None = None,
) -> tuple[HbondMetrics, Literal["generic", "csp_literature_v1"]]:
    diagnostics = compute_selector_hbond_diagnostics(
        mol=mol,
        selector=selector,
        max_distance_A=max_distance_A,
        neighbor_window=neighbor_window,
        min_donor_angle_deg=min_donor_angle_deg,
        min_acceptor_angle_deg=min_acceptor_angle_deg,
        box_vectors_A=box_vectors_A,
        periodic=periodic,
        connectivity_policy=connectivity_policy,
        requested_sites=requested_sites,
    )
    return diagnostics.metrics, diagnostics.applied_policy


def compute_selector_hbond_diagnostics(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    *,
    max_distance_A: float = 3.3,
    neighbor_window: int = 1,
    min_donor_angle_deg: float = 100.0,
    min_acceptor_angle_deg: float = 90.0,
    box_vectors_A: tuple[float, float, float] | None = None,
    periodic: bool | None = None,
    connectivity_policy: HbondConnectivityPolicy = "generic",
    requested_sites: Iterable[str] | None = None,
) -> SelectorHbondDiagnostics:
    applied_policy = resolve_hbond_connectivity_policy(
        mol,
        selector,
        requested_policy=connectivity_policy,
        requested_sites=requested_sites,
    )
    if applied_policy == "csp_literature_v1":
        target_edges = _build_csp_target_hbond_edges(
            mol,
            requested_sites=requested_sites,
        )
        if target_edges:
            metrics, family_metrics = _compute_target_hbond_metrics(
                mol,
                target_edges=target_edges,
                max_distance_A=max_distance_A,
                min_donor_angle_deg=min_donor_angle_deg,
                min_acceptor_angle_deg=min_acceptor_angle_deg,
            )
            return SelectorHbondDiagnostics(
                metrics=metrics,
                applied_policy=applied_policy,
                family_metrics=family_metrics,
            )
    return SelectorHbondDiagnostics(
        metrics=compute_hbond_metrics(
            mol=mol,
            selector=selector,
            max_distance_A=max_distance_A,
            neighbor_window=neighbor_window,
            min_donor_angle_deg=min_donor_angle_deg,
            min_acceptor_angle_deg=min_acceptor_angle_deg,
            box_vectors_A=box_vectors_A,
            periodic=periodic,
        ),
        applied_policy="generic",
        family_metrics={},
    )


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
