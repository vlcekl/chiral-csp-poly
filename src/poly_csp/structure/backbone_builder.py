from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.cache_versions import (
    BACKBONE_POSE_CACHE_SCHEMA_VERSION,
    BACKBONE_POSE_MODEL_VERSION,
)
from poly_csp.config.schema import HelixSpec
from poly_csp.structure.matrix import ScrewTransform
from poly_csp.structure.naming import AtomManifestEntry, build_atom_manifest
from poly_csp.structure.pbc import get_box_vectors_A
from poly_csp.structure.templates import (
    ExplicitResidueTemplate,
    build_residue_variant,
    load_explicit_backbone_template,
)
from poly_csp.topology.monomers import GlucoseMonomerTemplate, make_glucose_template
from poly_csp.topology.residue_state import (
    ResidueTemplateState,
    resolve_residue_template_states,
)
from poly_csp.topology.utils import (
    residue_label_maps,
    set_json_prop,
    set_residue_label_maps,
)


_RING_LABELS = ("C1", "C2", "C3", "C4", "C5", "O5")
_BOND_SIGMA_A = 0.05
_ANGLE_SIGMA_DEG = 8.0
_CLASH_CUTOFF_A = 1.55
_POSE_RADIUS_BOUNDS = (0.8, 2.4)
_POSE_TILT_BOUNDS = (-1.4, 1.4)
_POSE_PHASE_BOUNDS = (-math.pi, math.pi)
_PERIODIC_COMMENSURABILITY_TOL_RAD = 1e-4
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKBONE_POSE_CACHE_DIR = _REPO_ROOT / ".cache" / "poly_csp" / "backbone_pose"


@dataclass(frozen=True)
class BackboneBuildResult:
    """Structure-domain result for the canonical explicit-H backbone builder."""

    mol: Chem.Mol
    residue_maps: list[dict[str, int]]
    manifest: list[AtomManifestEntry]
    residue_states: list[ResidueTemplateState]
    pose_cache_summary: "BackbonePoseCacheSummary"


@dataclass(frozen=True)
class BackbonePose:
    radius_A: float
    tilt_x_rad: float
    tilt_y_rad: float
    phase_z_rad: float
    flip_ring_normal: bool = False


@dataclass(frozen=True)
class BackbonePoseCacheSummary:
    kind: str
    hit: bool
    cache_dir: str
    entry_path: str
    persisted: bool
    schema_version: int
    model_version: int


@dataclass(frozen=True)
class LinkageTargets:
    bond_length_A: float
    donor_angle_deg: float
    acceptor_angle_deg: float
    acceptor_angle_c2_deg: float


@dataclass(frozen=True)
class BackbonePoseEvaluation:
    score: float
    bond_length_A: float
    donor_angle_deg: float
    acceptor_angle_deg: float
    acceptor_angle_c2_deg: float
    min_inter_residue_distance_A: float


@dataclass(frozen=True)
class BackboneLinkageMetrics:
    donor_residue_index: int
    acceptor_residue_index: int
    bond_length_A: float
    donor_angle_deg: float
    acceptor_angle_deg: float
    acceptor_angle_c2_deg: float
    o4_h1_distance_A: float | None


def _copy_atom(atom: Chem.Atom) -> Chem.Atom:
    out = Chem.Atom(atom)
    out.SetNoImplicit(True)
    out.SetNumExplicitHs(0)
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / norm


def _orthonormal_basis(main_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = _normalize(main_dir)
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(u, trial))) > 0.85:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    n = _normalize(np.cross(u, trial))
    b = _normalize(np.cross(u, n))
    return u, n, b


def _choose_attachment_direction(
    coords: np.ndarray,
    anchor_idx: int,
    excluded: set[int],
    bond_len: float,
) -> np.ndarray:
    anchor_pos = np.asarray(coords[int(anchor_idx)], dtype=float)
    pieces: list[np.ndarray] = []
    for idx, point in enumerate(coords):
        if idx == anchor_idx or idx not in excluded:
            continue
        vec = anchor_pos - point
        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            pieces.append(vec / norm)
    base = _normalize(np.sum(np.asarray(pieces, dtype=float), axis=0)) if pieces else np.array(
        [1.0, 0.0, 0.0],
        dtype=float,
    )
    u, n, b = _orthonormal_basis(base)
    candidates = [
        u,
        -u,
        n,
        -n,
        b,
        -b,
        _normalize(u + n),
        _normalize(u - n),
        _normalize(u + b),
        _normalize(u - b),
    ]

    best = candidates[0]
    best_score = float("-inf")
    for direction in candidates:
        trial = anchor_pos + float(bond_len) * direction
        dmin = float("inf")
        for idx, point in enumerate(coords):
            if idx in excluded:
                continue
            dmin = min(dmin, float(np.linalg.norm(trial - point)))
        if dmin > best_score:
            best_score = dmin
            best = direction
    return _normalize(best)


def _tetrahedral_directions(back_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, n, b = _orthonormal_basis(back_dir)
    radial = 2.0 * np.sqrt(2.0) / 3.0
    dirs: list[np.ndarray] = []
    for angle in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
        direction = (
            (-1.0 / 3.0) * u
            + radial * (np.cos(angle) * n + np.sin(angle) * b)
        )
        dirs.append(_normalize(direction))
    return dirs[0], dirs[1], dirs[2]


def _bond_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-12:
        raise ValueError("Cannot measure angle with a near-zero bond vector.")
    cos_theta = float(np.dot(v1, v2) / denom)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _rotation_x(theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rotation_y(theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _wrap_angle(theta_rad: float) -> float:
    wrapped = (float(theta_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped)


def _validate_periodic_helix(topology_mol: Chem.Mol, helix_spec: HelixSpec) -> None:
    end_mode = (
        str(topology_mol.GetProp("_poly_csp_end_mode"))
        if topology_mol.HasProp("_poly_csp_end_mode")
        else "open"
    ).strip().lower()
    if end_mode != "periodic":
        return
    dp = int(topology_mol.GetIntProp("_poly_csp_dp")) if topology_mol.HasProp("_poly_csp_dp") else 0
    if dp < 2:
        raise ValueError("Periodic backbone construction requires dp >= 2.")
    total_rotation = _wrap_angle(float(helix_spec.theta_rad) * float(dp))
    if abs(total_rotation) > _PERIODIC_COMMENSURABILITY_TOL_RAD:
        raise ValueError(
            "Periodic end mode requires a helix/DP combination that closes the screw "
            "rotation over the simulation cell."
        )


def _minimum_image_delta_A(
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
        out[axis] -= length * np.round(out[axis] / length)
    return out


def _explicit_template_coords(template: ExplicitResidueTemplate) -> np.ndarray:
    conf = template.mol.GetConformer(0)
    return np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))


def _glucose_template_coords(template: GlucoseMonomerTemplate) -> np.ndarray:
    conf = template.mol.GetConformer(0)
    return np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))


def _ring_centroid(coords: np.ndarray, heavy_label_to_idx: dict[str, int]) -> np.ndarray:
    ring_indices = [heavy_label_to_idx[label] for label in _RING_LABELS]
    return coords[ring_indices].mean(axis=0)


def _ring_normal(coords: np.ndarray, heavy_label_to_idx: dict[str, int]) -> np.ndarray:
    ring_indices = [heavy_label_to_idx[label] for label in _RING_LABELS]
    ring_coords = coords[ring_indices]
    centered = ring_coords - ring_coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return _normalize(vh[-1])


def _chemical_frame_matrix(
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    *,
    flip_ring_normal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    origin = _ring_centroid(coords, heavy_label_to_idx)
    normal = _ring_normal(coords, heavy_label_to_idx)
    if flip_ring_normal:
        normal = -normal
    x_axis = coords[heavy_label_to_idx["C1"]] - coords[heavy_label_to_idx["O5"]]
    ex = _normalize(x_axis)
    ez = _normalize(normal - ex * float(np.dot(normal, ex)))
    ey = _normalize(np.cross(ez, ex))
    ez = _normalize(np.cross(ex, ey))
    basis = np.stack([ex, ey, ez], axis=1)
    return origin, basis


def _apply_backbone_pose(
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    pose: BackbonePose,
) -> np.ndarray:
    origin, basis = _chemical_frame_matrix(
        coords,
        heavy_label_to_idx,
        flip_ring_normal=pose.flip_ring_normal,
    )
    chemical_frame = (np.asarray(coords, dtype=float) - origin) @ basis
    rot = (
        ScrewTransform(theta_rad=pose.phase_z_rad, rise_A=0.0).matrix(1)[0]
        @ _rotation_y(pose.tilt_y_rad)
        @ _rotation_x(pose.tilt_x_rad)
    )
    return chemical_frame @ rot.T + np.array([pose.radius_A, 0.0, 0.0], dtype=float)


def _linkage_targets(
    polymer: str,
    representation: str,
) -> LinkageTargets:
    donor_template = load_explicit_backbone_template(
        polymer=polymer,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
    )
    donor_coords = _explicit_template_coords(donor_template)
    o4_h_indices = sorted(
        atom_idx
        for atom_idx, parent_label in donor_template.hydrogen_parent_label.items()
        if parent_label == "O4"
    )
    if not o4_h_indices:
        raise ValueError("Explicit backbone template is missing the O4 hydroxyl hydrogen.")
    donor_angle = _bond_angle_deg(
        donor_coords[donor_template.heavy_label_to_idx["C4"]],
        donor_coords[donor_template.heavy_label_to_idx["O4"]],
        donor_coords[o4_h_indices[0]],
    )

    acceptor_template = load_explicit_backbone_template(
        polymer=polymer,  # type: ignore[arg-type]
        representation="natural_oh",
    )
    acceptor_coords = _explicit_template_coords(acceptor_template)
    if "O1" not in acceptor_template.heavy_label_to_idx:
        raise ValueError("Natural glucose template must include O1 for linkage targets.")
    bond_length = float(
        np.linalg.norm(
            acceptor_coords[acceptor_template.heavy_label_to_idx["C1"]]
            - acceptor_coords[acceptor_template.heavy_label_to_idx["O1"]]
        )
    )
    acceptor_angle = _bond_angle_deg(
        acceptor_coords[acceptor_template.heavy_label_to_idx["O5"]],
        acceptor_coords[acceptor_template.heavy_label_to_idx["C1"]],
        acceptor_coords[acceptor_template.heavy_label_to_idx["O1"]],
    )
    acceptor_angle_c2 = _bond_angle_deg(
        acceptor_coords[acceptor_template.heavy_label_to_idx["C2"]],
        acceptor_coords[acceptor_template.heavy_label_to_idx["C1"]],
        acceptor_coords[acceptor_template.heavy_label_to_idx["O1"]],
    )
    return LinkageTargets(
        bond_length_A=bond_length,
        donor_angle_deg=donor_angle,
        acceptor_angle_deg=acceptor_angle,
        acceptor_angle_c2_deg=acceptor_angle_c2,
    )


def _evaluate_backbone_pose(
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    pose: BackbonePose,
    screw: ScrewTransform,
    targets: LinkageTargets,
) -> BackbonePoseEvaluation:
    residue0 = _apply_backbone_pose(coords, heavy_label_to_idx, pose)
    residue1 = screw.apply(residue0, 1)

    o4_idx = heavy_label_to_idx["O4"]
    c4_idx = heavy_label_to_idx["C4"]
    c1_idx = heavy_label_to_idx["C1"]
    o5_idx = heavy_label_to_idx["O5"]
    c2_idx = heavy_label_to_idx["C2"]

    bond_length = float(np.linalg.norm(residue0[o4_idx] - residue1[c1_idx]))
    donor_angle = _bond_angle_deg(residue0[c4_idx], residue0[o4_idx], residue1[c1_idx])
    acceptor_angle = _bond_angle_deg(residue0[o4_idx], residue1[c1_idx], residue1[o5_idx])
    acceptor_angle_c2 = _bond_angle_deg(
        residue0[o4_idx],
        residue1[c1_idx],
        residue1[c2_idx],
    )

    min_distance = float("inf")
    for idx0, point0 in enumerate(residue0):
        for idx1, point1 in enumerate(residue1):
            if idx0 == o4_idx and idx1 == c1_idx:
                continue
            min_distance = min(min_distance, float(np.linalg.norm(point0 - point1)))

    clash_penalty = 0.0
    if min_distance < _CLASH_CUTOFF_A:
        clash_penalty = (float(_CLASH_CUTOFF_A) - min_distance) * 10.0

    score = (
        ((bond_length - targets.bond_length_A) / _BOND_SIGMA_A) ** 2
        + ((donor_angle - targets.donor_angle_deg) / _ANGLE_SIGMA_DEG) ** 2
        + ((acceptor_angle - targets.acceptor_angle_deg) / _ANGLE_SIGMA_DEG) ** 2
        + ((acceptor_angle_c2 - targets.acceptor_angle_c2_deg) / _ANGLE_SIGMA_DEG) ** 2
        + clash_penalty
    )
    return BackbonePoseEvaluation(
        score=float(score),
        bond_length_A=bond_length,
        donor_angle_deg=donor_angle,
        acceptor_angle_deg=acceptor_angle,
        acceptor_angle_c2_deg=acceptor_angle_c2,
        min_inter_residue_distance_A=min_distance,
    )


def inspect_backbone_linkages(mol: Chem.Mol) -> list[BackboneLinkageMetrics]:
    """Return per-linkage geometry metrics for a built all-atom backbone structure."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Backbone linkage inspection requires 3D coordinates.")

    maps = residue_label_maps(mol)
    if len(maps) < 2:
        return []

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    linkage_pairs = [(residue_index, residue_index + 1) for residue_index in range(len(maps) - 1)]
    end_mode = str(mol.GetProp("_poly_csp_end_mode")) if mol.HasProp("_poly_csp_end_mode") else "open"
    box_vectors_A = get_box_vectors_A(mol)
    if end_mode == "periodic" and len(maps) > 1:
        linkage_pairs.append((len(maps) - 1, 0))

    out: list[BackboneLinkageMetrics] = []
    for donor_residue_index, acceptor_residue_index in linkage_pairs:
        donor = maps[donor_residue_index]
        acceptor = maps[acceptor_residue_index]
        for label in ("C4", "O4"):
            if label not in donor:
                raise ValueError(
                    f"Residue {donor_residue_index} is missing required donor label {label!r}."
                )
        for label in ("C1", "C2", "O5"):
            if label not in acceptor:
                raise ValueError(
                    f"Residue {acceptor_residue_index} is missing required acceptor label {label!r}."
                )

        donor_o4 = int(donor["O4"])
        acceptor_c1 = int(acceptor["C1"])
        donor_o4_xyz = np.asarray(xyz[donor_o4], dtype=float)
        acceptor_shift = np.zeros(3, dtype=float)
        if (
            end_mode == "periodic"
            and donor_residue_index == len(maps) - 1
            and acceptor_residue_index == 0
            and box_vectors_A is not None
        ):
            acceptor_delta = _minimum_image_delta_A(
                np.asarray(xyz[acceptor_c1], dtype=float) - donor_o4_xyz,
                box_vectors_A,
            )
            acceptor_shift = donor_o4_xyz + acceptor_delta - np.asarray(
                xyz[acceptor_c1],
                dtype=float,
            )

        donor_angle = _bond_angle_deg(
            xyz[int(donor["C4"])],
            donor_o4_xyz,
            np.asarray(xyz[acceptor_c1], dtype=float) + acceptor_shift,
        )
        acceptor_angle = _bond_angle_deg(
            donor_o4_xyz,
            np.asarray(xyz[acceptor_c1], dtype=float) + acceptor_shift,
            np.asarray(xyz[int(acceptor["O5"])], dtype=float) + acceptor_shift,
        )
        acceptor_angle_c2 = _bond_angle_deg(
            donor_o4_xyz,
            np.asarray(xyz[acceptor_c1], dtype=float) + acceptor_shift,
            np.asarray(xyz[int(acceptor["C2"])], dtype=float) + acceptor_shift,
        )

        c1_atom = mol.GetAtomWithIdx(acceptor_c1)
        h1_neighbors = [int(nbr.GetIdx()) for nbr in c1_atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
        if len(h1_neighbors) > 1:
            raise ValueError(
                f"Residue {acceptor_residue_index} has more than one hydrogen bonded to C1."
            )
        o4_h1_distance = (
            None
            if not h1_neighbors
            else float(
                np.linalg.norm(
                    donor_o4_xyz
                    - (np.asarray(xyz[h1_neighbors[0]], dtype=float) + acceptor_shift)
                )
            )
        )

        out.append(
            BackboneLinkageMetrics(
                donor_residue_index=donor_residue_index,
                acceptor_residue_index=acceptor_residue_index,
                bond_length_A=float(
                    np.linalg.norm(
                        donor_o4_xyz
                        - (np.asarray(xyz[acceptor_c1], dtype=float) + acceptor_shift)
                    )
                ),
                donor_angle_deg=donor_angle,
                acceptor_angle_deg=acceptor_angle,
                acceptor_angle_c2_deg=acceptor_angle_c2,
                o4_h1_distance_A=o4_h1_distance,
            )
        )
    return out


def _candidate_backbone_poses() -> list[BackbonePose]:
    radii = (0.8, 1.2, 1.6, 2.0)
    tilt_x = (0.0, 0.35, -0.35)
    tilt_y = (0.0, 0.7, -0.7, 1.1, -1.1)
    phase_z = (0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0)
    candidates: list[BackbonePose] = []
    for flip in (False, True):
        for radius in radii:
            for ax in tilt_x:
                for ay in tilt_y:
                    for az in phase_z:
                        candidates.append(
                            BackbonePose(
                                radius_A=float(radius),
                                tilt_x_rad=float(ax),
                                tilt_y_rad=float(ay),
                                phase_z_rad=float(az),
                                flip_ring_normal=flip,
                            )
                        )
    return candidates


def _clip_pose(pose: BackbonePose) -> BackbonePose:
    return BackbonePose(
        radius_A=float(np.clip(pose.radius_A, *_POSE_RADIUS_BOUNDS)),
        tilt_x_rad=float(np.clip(pose.tilt_x_rad, *_POSE_TILT_BOUNDS)),
        tilt_y_rad=float(np.clip(pose.tilt_y_rad, *_POSE_TILT_BOUNDS)),
        phase_z_rad=_wrap_angle(pose.phase_z_rad),
        flip_ring_normal=pose.flip_ring_normal,
    )


def _refine_backbone_pose(
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    screw: ScrewTransform,
    targets: LinkageTargets,
    seed: BackbonePose,
) -> tuple[BackbonePose, BackbonePoseEvaluation]:
    best_pose = _clip_pose(seed)
    best_eval = _evaluate_backbone_pose(coords, heavy_label_to_idx, best_pose, screw, targets)
    schedules = (
        (0.4, 0.3, 0.3),
        (0.2, 0.15, 0.15),
        (0.1, 0.08, 0.08),
        (0.05, 0.04, 0.04),
        (0.02, 0.02, 0.02),
    )
    for radius_step, tilt_step, phase_step in schedules:
        improved = True
        while improved:
            improved = False
            candidates = (
                BackbonePose(
                    radius_A=best_pose.radius_A + delta,
                    tilt_x_rad=best_pose.tilt_x_rad,
                    tilt_y_rad=best_pose.tilt_y_rad,
                    phase_z_rad=best_pose.phase_z_rad,
                    flip_ring_normal=best_pose.flip_ring_normal,
                )
                for delta in (-radius_step, radius_step)
            )
            angle_candidates = [
                BackbonePose(
                    radius_A=best_pose.radius_A,
                    tilt_x_rad=best_pose.tilt_x_rad + delta,
                    tilt_y_rad=best_pose.tilt_y_rad,
                    phase_z_rad=best_pose.phase_z_rad,
                    flip_ring_normal=best_pose.flip_ring_normal,
                )
                for delta in (-tilt_step, tilt_step)
            ]
            angle_candidates.extend(
                BackbonePose(
                    radius_A=best_pose.radius_A,
                    tilt_x_rad=best_pose.tilt_x_rad,
                    tilt_y_rad=best_pose.tilt_y_rad + delta,
                    phase_z_rad=best_pose.phase_z_rad,
                    flip_ring_normal=best_pose.flip_ring_normal,
                )
                for delta in (-tilt_step, tilt_step)
            )
            angle_candidates.extend(
                BackbonePose(
                    radius_A=best_pose.radius_A,
                    tilt_x_rad=best_pose.tilt_x_rad,
                    tilt_y_rad=best_pose.tilt_y_rad,
                    phase_z_rad=best_pose.phase_z_rad + delta,
                    flip_ring_normal=best_pose.flip_ring_normal,
                )
                for delta in (-phase_step, phase_step)
            )
            for candidate in list(candidates) + angle_candidates:
                clipped = _clip_pose(candidate)
                evaluation = _evaluate_backbone_pose(
                    coords, heavy_label_to_idx, clipped, screw, targets
                )
                if evaluation.score + 1e-12 < best_eval.score:
                    best_pose = clipped
                    best_eval = evaluation
                    improved = True
    return best_pose, best_eval


_BACKBONE_POSE_CACHE: dict[tuple[str, str, float, float], BackbonePose] = {}


def _backbone_pose_cache_identity(
    polymer: str,
    representation: str,
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    helix_spec: HelixSpec,
) -> tuple[str, dict[str, object]]:
    coords_array = np.asarray(coords, dtype=np.float64)
    identity = {
        "schema_version": BACKBONE_POSE_CACHE_SCHEMA_VERSION,
        "model_version": BACKBONE_POSE_MODEL_VERSION,
        "kind": "backbone_pose",
        "polymer": str(polymer),
        "representation": str(representation),
        "theta_rad": float(helix_spec.theta_rad),
        "rise_A": float(helix_spec.rise_A),
        "coords_sha256": hashlib.sha256(coords_array.tobytes()).hexdigest(),
        "heavy_label_to_idx": {
            str(label): int(idx)
            for label, idx in sorted(heavy_label_to_idx.items())
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24], identity


def _backbone_pose_cache_entry(
    polymer: str,
    representation: str,
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    helix_spec: HelixSpec,
) -> tuple[Path, dict[str, object]]:
    key, identity = _backbone_pose_cache_identity(
        polymer=polymer,
        representation=representation,
        coords=coords,
        heavy_label_to_idx=heavy_label_to_idx,
        helix_spec=helix_spec,
    )
    return (
        _BACKBONE_POSE_CACHE_DIR
        / str(polymer).lower()
        / str(representation).lower()
        / key
        / "pose.json",
        identity,
    )


def _backbone_pose_cache_summary(
    *,
    kind: str,
    payload_path: Path,
    persisted: bool,
) -> BackbonePoseCacheSummary:
    return BackbonePoseCacheSummary(
        kind=str(kind),
        hit=bool(kind in {"memory", "disk"}),
        cache_dir=str(_BACKBONE_POSE_CACHE_DIR),
        entry_path=str(payload_path),
        persisted=bool(persisted),
        schema_version=BACKBONE_POSE_CACHE_SCHEMA_VERSION,
        model_version=BACKBONE_POSE_MODEL_VERSION,
    )


def _load_disk_cached_backbone_pose(
    polymer: str,
    representation: str,
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    helix_spec: HelixSpec,
) -> BackbonePose | None:
    payload_path, expected_identity = _backbone_pose_cache_entry(
        polymer=polymer,
        representation=representation,
        coords=coords,
        heavy_label_to_idx=heavy_label_to_idx,
        helix_spec=helix_spec,
    )
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != BACKBONE_POSE_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("model_version") != BACKBONE_POSE_MODEL_VERSION:
        return None
    if payload.get("identity") != expected_identity:
        return None
    pose_payload = payload.get("pose")
    if not isinstance(pose_payload, dict):
        return None
    try:
        return BackbonePose(
            radius_A=float(pose_payload["radius_A"]),
            tilt_x_rad=float(pose_payload["tilt_x_rad"]),
            tilt_y_rad=float(pose_payload["tilt_y_rad"]),
            phase_z_rad=float(pose_payload["phase_z_rad"]),
            flip_ring_normal=bool(pose_payload["flip_ring_normal"]),
        )
    except Exception:
        return None


def _store_disk_cached_backbone_pose(
    polymer: str,
    representation: str,
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    helix_spec: HelixSpec,
    pose: BackbonePose,
) -> bool:
    payload_path, identity = _backbone_pose_cache_entry(
        polymer=polymer,
        representation=representation,
        coords=coords,
        heavy_label_to_idx=heavy_label_to_idx,
        helix_spec=helix_spec,
    )
    payload = {
        "schema_version": BACKBONE_POSE_CACHE_SCHEMA_VERSION,
        "model_version": BACKBONE_POSE_MODEL_VERSION,
        "identity": identity,
        "pose": {
            "radius_A": float(pose.radius_A),
            "tilt_x_rad": float(pose.tilt_x_rad),
            "tilt_y_rad": float(pose.tilt_y_rad),
            "phase_z_rad": float(pose.phase_z_rad),
            "flip_ring_normal": bool(pose.flip_ring_normal),
        },
    }
    try:
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def _fit_backbone_pose(
    polymer: str,
    representation: str,
    coords: np.ndarray,
    heavy_label_to_idx: dict[str, int],
    helix_spec: HelixSpec,
) -> tuple[BackbonePose, BackbonePoseCacheSummary]:
    cache_key = (
        str(polymer),
        str(representation),
        float(helix_spec.theta_rad),
        float(helix_spec.rise_A),
    )
    payload_path, _ = _backbone_pose_cache_entry(
        polymer=polymer,
        representation=representation,
        coords=coords,
        heavy_label_to_idx=heavy_label_to_idx,
        helix_spec=helix_spec,
    )
    cached = _BACKBONE_POSE_CACHE.get(cache_key)
    if cached is not None:
        return (
            cached,
            _backbone_pose_cache_summary(
                kind="memory",
                payload_path=payload_path,
                persisted=payload_path.exists(),
            ),
        )

    disk_cached = _load_disk_cached_backbone_pose(
        polymer=polymer,
        representation=representation,
        coords=coords,
        heavy_label_to_idx=heavy_label_to_idx,
        helix_spec=helix_spec,
    )
    if disk_cached is not None:
        _BACKBONE_POSE_CACHE[cache_key] = disk_cached
        return (
            disk_cached,
            _backbone_pose_cache_summary(
                kind="disk",
                payload_path=payload_path,
                persisted=True,
            ),
        )

    if abs(float(helix_spec.theta_rad)) < 1e-10 and abs(float(helix_spec.rise_A)) < 1e-10:
        neutral = BackbonePose(
            radius_A=0.0,
            tilt_x_rad=0.0,
            tilt_y_rad=0.0,
            phase_z_rad=0.0,
            flip_ring_normal=False,
        )
        _BACKBONE_POSE_CACHE[cache_key] = neutral
        persisted = _store_disk_cached_backbone_pose(
            polymer=polymer,
            representation=representation,
            coords=coords,
            heavy_label_to_idx=heavy_label_to_idx,
            helix_spec=helix_spec,
            pose=neutral,
        )
        return (
            neutral,
            _backbone_pose_cache_summary(
                kind="build",
                payload_path=payload_path,
                persisted=persisted,
            ),
        )

    screw = ScrewTransform(theta_rad=helix_spec.theta_rad, rise_A=helix_spec.rise_A)
    targets = _linkage_targets(polymer=polymer, representation=representation)
    scored_seeds = sorted(
        (
            (
                _evaluate_backbone_pose(coords, heavy_label_to_idx, pose, screw, targets).score,
                pose,
            )
            for pose in _candidate_backbone_poses()
        ),
        key=lambda item: item[0],
    )

    best_pose: BackbonePose | None = None
    best_eval: BackbonePoseEvaluation | None = None
    for _, seed in scored_seeds[:12]:
        pose, evaluation = _refine_backbone_pose(
            coords,
            heavy_label_to_idx,
            screw,
            targets,
            seed,
        )
        if best_eval is None or evaluation.score < best_eval.score:
            best_pose = pose
            best_eval = evaluation

    if best_pose is None or best_eval is None:
        raise RuntimeError("Failed to fit a helical backbone pose.")
    if abs(best_eval.bond_length_A - targets.bond_length_A) > 0.2:
        raise ValueError(
            "Helix parameters are incompatible with a chemically plausible glycosidic bond."
        )
    _BACKBONE_POSE_CACHE[cache_key] = best_pose
    persisted = _store_disk_cached_backbone_pose(
        polymer=polymer,
        representation=representation,
        coords=coords,
        heavy_label_to_idx=heavy_label_to_idx,
        helix_spec=helix_spec,
        pose=best_pose,
    )
    return (
        best_pose,
        _backbone_pose_cache_summary(
            kind="build",
            payload_path=payload_path,
            persisted=persisted,
        ),
    )


def build_backbone_heavy_coords(
    template: GlucoseMonomerTemplate,
    helix_spec: HelixSpec,
    dp: int,
) -> np.ndarray:
    """Return the heavy-atom coordinates implied by the canonical backbone pose solver."""
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}")

    coords = _glucose_template_coords(template)
    pose, _ = _fit_backbone_pose(
        polymer=template.polymer,
        representation=template.representation,
        coords=coords,
        heavy_label_to_idx=template.atom_idx,
        helix_spec=helix_spec,
    )
    residue0 = _apply_backbone_pose(coords, template.atom_idx, pose)
    screw = ScrewTransform(theta_rad=helix_spec.theta_rad, rise_A=helix_spec.rise_A)
    return np.concatenate([screw.apply(residue0, i) for i in range(dp)], axis=0)


def _append_atom(
    rw: Chem.RWMol,
    positions: list[np.ndarray],
    atom: Chem.Atom,
    position: np.ndarray,
) -> int:
    atom_idx = rw.AddAtom(atom)
    positions.append(np.asarray(position, dtype=float))
    return int(atom_idx)


def _append_cap_heavy_atom(
    rw: Chem.RWMol,
    positions: list[np.ndarray],
    atomic_num: int,
    position: np.ndarray,
    *,
    side: str,
) -> int:
    atom = Chem.Atom(int(atomic_num))
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetProp("_poly_csp_component", "backbone")
    atom.SetProp("_poly_csp_terminal_cap_side", side)
    return _append_atom(rw, positions, atom, position)


def _append_cap_hydrogen(
    rw: Chem.RWMol,
    positions: list[np.ndarray],
    parent_idx: int,
    position: np.ndarray,
    *,
    side: str,
) -> int:
    atom = Chem.Atom(1)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetIntProp("_poly_csp_parent_heavy_idx", int(parent_idx))
    atom.SetProp("_poly_csp_component", "backbone")
    atom.SetProp("_poly_csp_terminal_cap_side", side)
    atom_idx = _append_atom(rw, positions, atom, position)
    rw.AddBond(int(parent_idx), int(atom_idx), Chem.BondType.SINGLE)
    return atom_idx


def _append_methyl_cap(
    rw: Chem.RWMol,
    positions: list[np.ndarray],
    coords: np.ndarray,
    anchor_idx: int,
    *,
    side: str,
) -> list[int]:
    anchor = rw.GetAtomWithIdx(int(anchor_idx))
    bond_len = 1.43 if anchor.GetAtomicNum() == 8 else 1.50
    excluded = {int(anchor_idx)}
    excluded.update(int(nbr.GetIdx()) for nbr in anchor.GetNeighbors())
    direction = _choose_attachment_direction(coords, anchor_idx, excluded, bond_len)
    anchor_pos = np.asarray(coords[int(anchor_idx)], dtype=float)
    carbon_pos = anchor_pos + bond_len * direction
    carbon_idx = _append_cap_heavy_atom(
        rw,
        positions,
        atomic_num=6,
        position=carbon_pos,
        side=side,
    )
    rw.AddBond(int(anchor_idx), int(carbon_idx), Chem.BondType.SINGLE)

    h_dirs = _tetrahedral_directions(anchor_pos - carbon_pos)
    for h_dir in h_dirs:
        _append_cap_hydrogen(
            rw,
            positions,
            carbon_idx,
            carbon_pos + 1.09 * h_dir,
            side=side,
        )
    return [carbon_idx]


def _append_hydroxyl_cap(
    rw: Chem.RWMol,
    positions: list[np.ndarray],
    coords: np.ndarray,
    anchor_idx: int,
    *,
    side: str,
) -> list[int]:
    anchor = rw.GetAtomWithIdx(int(anchor_idx))
    if anchor.GetAtomicNum() != 6:
        raise ValueError("Hydroxyl cap requires a carbon anchor.")
    excluded = {int(anchor_idx)}
    excluded.update(int(nbr.GetIdx()) for nbr in anchor.GetNeighbors())
    direction = _choose_attachment_direction(coords, anchor_idx, excluded, 1.43)
    anchor_pos = np.asarray(coords[int(anchor_idx)], dtype=float)
    oxygen_pos = anchor_pos + 1.43 * direction
    oxygen_idx = _append_cap_heavy_atom(
        rw,
        positions,
        atomic_num=8,
        position=oxygen_pos,
        side=side,
    )
    rw.AddBond(int(anchor_idx), int(oxygen_idx), Chem.BondType.SINGLE)
    _append_cap_hydrogen(
        rw,
        positions,
        oxygen_idx,
        oxygen_pos + 0.96 * direction,
        side=side,
    )
    return [oxygen_idx]


def _append_acetyl_cap(
    rw: Chem.RWMol,
    positions: list[np.ndarray],
    coords: np.ndarray,
    anchor_idx: int,
    *,
    side: str,
) -> list[int]:
    anchor = rw.GetAtomWithIdx(int(anchor_idx))
    bond_len = 1.43 if anchor.GetAtomicNum() == 8 else 1.50
    excluded = {int(anchor_idx)}
    excluded.update(int(nbr.GetIdx()) for nbr in anchor.GetNeighbors())
    direction = _choose_attachment_direction(coords, anchor_idx, excluded, bond_len)
    anchor_pos = np.asarray(coords[int(anchor_idx)], dtype=float)
    carbonyl_c_pos = anchor_pos + bond_len * direction
    carbonyl_c_idx = _append_cap_heavy_atom(
        rw,
        positions,
        atomic_num=6,
        position=carbonyl_c_pos,
        side=side,
    )
    rw.AddBond(int(anchor_idx), int(carbonyl_c_idx), Chem.BondType.SINGLE)

    u, n, _ = _orthonormal_basis(anchor_pos - carbonyl_c_pos)
    carbonyl_o_pos = carbonyl_c_pos + 1.23 * _normalize(0.5 * u + 0.866 * n)
    methyl_pos = carbonyl_c_pos + 1.52 * _normalize(0.5 * u - 0.866 * n)
    carbonyl_o_idx = _append_cap_heavy_atom(
        rw,
        positions,
        atomic_num=8,
        position=carbonyl_o_pos,
        side=side,
    )
    methyl_idx = _append_cap_heavy_atom(
        rw,
        positions,
        atomic_num=6,
        position=methyl_pos,
        side=side,
    )
    rw.AddBond(int(carbonyl_c_idx), int(carbonyl_o_idx), Chem.BondType.DOUBLE)
    rw.AddBond(int(carbonyl_c_idx), int(methyl_idx), Chem.BondType.SINGLE)

    h_dirs = _tetrahedral_directions(carbonyl_c_pos - methyl_pos)
    for h_dir in h_dirs:
        _append_cap_hydrogen(
            rw,
            positions,
            methyl_idx,
            methyl_pos + 1.09 * h_dir,
            side=side,
        )

    return [carbonyl_c_idx, carbonyl_o_idx, methyl_idx]


def _append_cap(
    rw: Chem.RWMol,
    positions: list[np.ndarray],
    anchor_idx: int,
    *,
    cap_name: str | None,
    side: str,
) -> list[int]:
    cap = "none" if cap_name is None else str(cap_name).strip().lower()
    if cap in {"", "none", "h", "hydrogen"}:
        return []

    coords = np.asarray(positions, dtype=float)
    if cap in {"methyl", "methoxy"}:
        return _append_methyl_cap(rw, positions, coords, anchor_idx, side=side)
    if cap in {"hydroxyl", "oh"}:
        return _append_hydroxyl_cap(rw, positions, coords, anchor_idx, side=side)
    if cap == "acetyl":
        return _append_acetyl_cap(rw, positions, coords, anchor_idx, side=side)
    raise ValueError(f"Unsupported cap type {cap_name!r}.")


def _set_backbone_atom_metadata(
    atom: Chem.Atom,
    residue_index: int,
    residue_label: str,
) -> None:
    atom.SetProp("_poly_csp_component", "backbone")
    atom.SetIntProp("_poly_csp_residue_index", int(residue_index))
    atom.SetProp("_poly_csp_residue_label", str(residue_label))
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.UpdatePropertyCache(strict=False)


def _set_backbone_hydrogen_metadata(
    atom: Chem.Atom,
    residue_index: int,
    parent_idx: int,
) -> None:
    atom.SetProp("_poly_csp_component", "backbone")
    atom.SetIntProp("_poly_csp_residue_index", int(residue_index))
    atom.SetIntProp("_poly_csp_parent_heavy_idx", int(parent_idx))
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.UpdatePropertyCache(strict=False)


def build_backbone_structure(
    topology_mol: Chem.Mol,
    helix_spec: HelixSpec,
) -> BackboneBuildResult:
    """Build the canonical structure-domain all-atom backbone from heavy topology metadata."""
    _validate_periodic_helix(topology_mol, helix_spec)
    residue_states = resolve_residue_template_states(topology_mol)
    if not residue_states:
        raise ValueError("Backbone construction requires at least one residue state.")

    screw = ScrewTransform(theta_rad=helix_spec.theta_rad, rise_A=helix_spec.rise_A)
    first_state = residue_states[0]
    pose_template = make_glucose_template(
        polymer=first_state.polymer,
        monomer_representation=first_state.representation,
    )
    backbone_pose, pose_cache_summary = _fit_backbone_pose(
        polymer=first_state.polymer,
        representation=first_state.representation,
        coords=_glucose_template_coords(pose_template),
        heavy_label_to_idx=pose_template.atom_idx,
        helix_spec=helix_spec,
    )
    rw = Chem.RWMol()
    positions: list[np.ndarray] = []
    residue_maps: list[dict[str, int]] = []
    residue_variants: list[ExplicitResidueTemplate] = []
    residue_variant_coords: list[np.ndarray] = []
    residue_local_to_global: list[dict[int, int]] = []

    for state in residue_states:
        base_template = load_explicit_backbone_template(
            polymer=state.polymer,
            representation=state.representation,
        )
        variant = build_residue_variant(base_template, state)
        residue_variants.append(variant)
        coords = screw.apply(
            _apply_backbone_pose(
                _explicit_template_coords(variant),
                variant.heavy_label_to_idx,
                backbone_pose,
            ),
            state.residue_index,
        )
        residue_variant_coords.append(coords)
        reverse_heavy = {
            int(local_idx): str(label)
            for label, local_idx in variant.heavy_label_to_idx.items()
        }

        local_to_global: dict[int, int] = {}
        residue_map: dict[str, int] = {}
        for local_idx, residue_label in sorted(reverse_heavy.items()):
            atom = _copy_atom(variant.mol.GetAtomWithIdx(int(local_idx)))
            atom_idx = _append_atom(rw, positions, atom, coords[int(local_idx)])
            local_to_global[int(local_idx)] = atom_idx
            residue_map[residue_label] = atom_idx
            _set_backbone_atom_metadata(
                rw.GetAtomWithIdx(int(atom_idx)),
                residue_index=state.residue_index,
                residue_label=residue_label,
            )

        for bond in variant.mol.GetBonds():
            begin = int(bond.GetBeginAtomIdx())
            end = int(bond.GetEndAtomIdx())
            if begin not in local_to_global or end not in local_to_global:
                continue
            rw.AddBond(
                int(local_to_global[begin]),
                int(local_to_global[end]),
                bond.GetBondType(),
            )

        residue_maps.append(residue_map)
        residue_local_to_global.append(local_to_global)

    for residue_index in range(1, len(residue_states)):
        prev_map = residue_maps[residue_index - 1]
        curr_map = residue_maps[residue_index]
        rw.AddBond(int(prev_map["O4"]), int(curr_map["C1"]), Chem.BondType.SINGLE)

    if residue_states[0].end_mode == "periodic" and len(residue_states) > 1:
        rw.AddBond(
            int(residue_maps[-1]["O4"]),
            int(residue_maps[0]["C1"]),
            Chem.BondType.SINGLE,
        )

    cap_indices: dict[str, list[int]] = {"left": [], "right": []}
    left_state = residue_states[0]
    if left_state.left_cap:
        if left_state.left_anchor_label is None:
            raise ValueError("Left cap is missing its anchor label.")
        left_anchor = residue_maps[0].get(left_state.left_anchor_label)
        if left_anchor is None:
            raise ValueError(
                f"Left cap anchor {left_state.left_anchor_label!r} is unavailable."
            )
        cap_indices["left"] = _append_cap(
            rw,
            positions,
            left_anchor,
            cap_name=left_state.left_cap,
            side="left",
        )

    right_state = residue_states[-1]
    if right_state.right_cap:
        if right_state.right_anchor_label is None:
            raise ValueError("Right cap is missing its anchor label.")
        right_anchor = residue_maps[-1].get(right_state.right_anchor_label)
        if right_anchor is None:
            raise ValueError(
                f"Right cap anchor {right_state.right_anchor_label!r} is unavailable."
            )
        cap_indices["right"] = _append_cap(
            rw,
            positions,
            right_anchor,
            cap_name=right_state.right_cap,
            side="right",
        )

    for state, variant, local_to_global, coords in zip(
        residue_states,
        residue_variants,
        residue_local_to_global,
        residue_variant_coords,
        strict=True,
    ):
        for local_idx, parent_label in sorted(variant.hydrogen_parent_label.items()):
            parent_local = int(variant.heavy_label_to_idx[parent_label])
            parent_global = int(local_to_global[parent_local])
            atom = _copy_atom(variant.mol.GetAtomWithIdx(int(local_idx)))
            atom_idx = _append_atom(rw, positions, atom, coords[int(local_idx)])
            rw.AddBond(int(parent_global), int(atom_idx), Chem.BondType.SINGLE)
            _set_backbone_hydrogen_metadata(
                rw.GetAtomWithIdx(int(atom_idx)),
                residue_index=state.residue_index,
                parent_idx=parent_global,
            )

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    conf = Chem.Conformer(out.GetNumAtoms())
    for atom_idx, position in enumerate(positions):
        x, y, z = position
        conf.SetAtomPosition(int(atom_idx), Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)

    out.SetIntProp("_poly_csp_dp", len(residue_states))
    out.SetProp("_poly_csp_polymer", first_state.polymer)
    out.SetProp("_poly_csp_representation", first_state.representation)
    out.SetProp("_poly_csp_end_mode", first_state.end_mode)
    out.SetProp("_poly_csp_helix_name", str(helix_spec.name))
    if helix_spec.repeat_residues is not None:
        out.SetIntProp("_poly_csp_helix_repeat_residues", int(helix_spec.repeat_residues))
    if helix_spec.repeat_turns is not None:
        out.SetIntProp("_poly_csp_helix_repeat_turns", int(helix_spec.repeat_turns))
    if helix_spec.axial_repeat_A is not None:
        out.SetDoubleProp("_poly_csp_helix_axial_repeat_A", float(helix_spec.axial_repeat_A))
    set_residue_label_maps(out, residue_maps)
    set_json_prop(out, "_poly_csp_removed_old_indices_json", [])
    set_json_prop(
        out,
        "_poly_csp_end_caps_json",
        {
            "left": left_state.left_cap or "none",
            "right": right_state.right_cap or "none",
        },
    )
    set_json_prop(
        out,
        "_poly_csp_terminal_meta_json",
        {
            "left_anchor_label": left_state.left_anchor_label,
            "left_anchor_idx": (
                residue_maps[0].get(left_state.left_anchor_label)
                if left_state.left_anchor_label is not None
                else None
            ),
            "right_anchor_label": right_state.right_anchor_label,
            "right_anchor_idx": (
                residue_maps[-1].get(right_state.right_anchor_label)
                if right_state.right_anchor_label is not None
                else None
            ),
        },
    )
    set_json_prop(out, "_poly_csp_terminal_cap_indices_json", cap_indices)
    out.SetBoolProp("_poly_csp_terminal_topology_pending", False)

    manifest = build_atom_manifest(out)
    return BackboneBuildResult(
        mol=out,
        residue_maps=residue_maps,
        manifest=manifest,
        residue_states=residue_states,
        pose_cache_summary=pose_cache_summary,
    )
