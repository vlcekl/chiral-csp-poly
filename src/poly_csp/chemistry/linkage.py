# poly_csp/chemistry/linkage.py
"""Ideal internal coordinates for selector–backbone linkages.

Each linkage type is defined by bond lengths, bond angles, and atom
identities for the bridging fragment.  These are used to place the
first few atoms of the selector in chemically correct geometry before
the aromatic ring is positioned via dihedral setting.

All distances in angstrom, angles in degrees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class LinkageGeometry:
    """Ideal geometry for a 3-atom bridging fragment A–B–C.

    The fragment connects the backbone oxygen (anchor) to the selector
    core.  For example, in a carbamate O–C(=O)–NH, the three atoms are:
      A = O_sugar (anchor, already placed)
      B = C_carbonyl
      C = N_amide (or next selector heavy atom)

    The ``sidechain_*`` fields describe the position of the =O in
    carbamate / ester fragments.
    """
    # A–B bond
    ab_bond_A: float
    # B–C bond
    bc_bond_A: float
    # A–B–C angle
    abc_angle_deg: float
    # Anchor–A–B angle (e.g. C_sugar–O–C_carbonyl)
    anchor_ab_angle_deg: float
    # Sidechain (e.g. =O on carbonyl): bond length and angle from A–B axis
    sidechain_bond_A: float
    sidechain_ab_angle_deg: float   # A–B=O angle


# Carbamate: Sugar–O–C(=O)–NH–Ar
CARBAMATE = LinkageGeometry(
    ab_bond_A=1.36,              # O_sugar – C_carbonyl
    bc_bond_A=1.36,              # C_carbonyl – N_amide
    abc_angle_deg=110.0,         # O–C–N (sp2 carbonyl)
    anchor_ab_angle_deg=117.0,   # C_sugar–O–C (ester/carbamate)
    sidechain_bond_A=1.22,       # C=O double bond
    sidechain_ab_angle_deg=125.0,  # O_sugar–C=O angle
)

# Ester: Sugar–O–C(=O)–O–Ar  (for future selector types like benzoates)
ESTER = LinkageGeometry(
    ab_bond_A=1.36,
    bc_bond_A=1.34,
    abc_angle_deg=111.0,
    anchor_ab_angle_deg=117.0,
    sidechain_bond_A=1.21,
    sidechain_ab_angle_deg=125.0,
)

# Ether: Sugar–O–CX2–Ar  (simplified single-bond linkage)
ETHER = LinkageGeometry(
    ab_bond_A=1.43,
    bc_bond_A=1.43,
    abc_angle_deg=112.0,
    anchor_ab_angle_deg=112.0,
    sidechain_bond_A=0.0,  # no sidechain
    sidechain_ab_angle_deg=0.0,
)


LINKAGE_TABLE = {
    "carbamate": CARBAMATE,
    "ester": ESTER,
    "ether": ETHER,
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _orthogonal_in_plane(
    main: np.ndarray, ref: np.ndarray,
) -> np.ndarray:
    """Return a unit vector in the plane of *main* and *ref*, perpendicular to *main*."""
    u = _normalize(main)
    component = ref - np.dot(ref, u) * u
    n = float(np.linalg.norm(component))
    if n < 1e-10:
        # ref is parallel to main, pick arbitrary perpendicular
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(trial, u))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        component = trial - np.dot(trial, u) * u
    return _normalize(component)


def build_linkage_coords(
    anchor_pos: np.ndarray,
    anchor_parent_pos: np.ndarray,
    geom: LinkageGeometry,
    plane_ref: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build ideal coordinates for the 3-atom linker fragment.

    Parameters
    ----------
    anchor_pos : (3,) array
        Position of atom A (e.g. O_sugar — already placed in the polymer).
    anchor_parent_pos : (3,) array
        Position of the atom bonded to A on the backbone side (e.g. C_sugar).
        Used to define the anchor–A–B angle.
    geom : LinkageGeometry
        Ideal internal coordinates for the linkage type.
    plane_ref : (3,) optional
        Reference direction to define the dihedral plane.  If None, an
        arbitrary consistent plane is chosen.

    Returns
    -------
    b_pos : (3,) array  — position of atom B (e.g. C_carbonyl)
    c_pos : (3,) array  — position of atom C (e.g. N_amide)
    sidechain_pos : (3,) or None — position of sidechain atom (e.g. =O)
    """
    anchor = np.asarray(anchor_pos, dtype=float)
    parent = np.asarray(anchor_parent_pos, dtype=float)

    # Direction from parent → anchor  (the bond entering the anchor)
    incoming = _normalize(anchor - parent)

    # Build B: extend from A along a direction that makes the correct
    # anchor–A–B angle with the incoming bond.
    angle_ab = np.deg2rad(geom.anchor_ab_angle_deg)
    perp = _orthogonal_in_plane(
        incoming,
        np.asarray(plane_ref, dtype=float) if plane_ref is not None
        else np.array([0.0, 0.0, 1.0]),
    )
    # Rotate incoming by (pi - angle_ab) around perp to get A→B direction
    ab_dir = (
        incoming * np.cos(np.pi - angle_ab)
        + perp * np.sin(np.pi - angle_ab)
    )
    ab_dir = _normalize(ab_dir)
    b_pos = anchor + geom.ab_bond_A * ab_dir

    # Build C: extend from B such that angle A–B–C = abc_angle_deg.
    # The angle is measured at B between vectors B→A and B→C.
    # We rotate ba_dir towards perp by the supplement of abc_angle_deg
    # because C is on the far side of atom B from A:
    #   B→C direction makes angle abc_angle_deg with B→A.
    angle_bc = np.deg2rad(geom.abc_angle_deg)
    ba_dir = _normalize(anchor - b_pos)
    perp_bc = _orthogonal_in_plane(ba_dir, perp)
    # cos(angle) with ba_dir, sin(angle) component in perp direction
    bc_dir = (
        ba_dir * np.cos(angle_bc)
        + perp_bc * np.sin(angle_bc)
    )
    bc_dir = _normalize(bc_dir)
    c_pos = b_pos + geom.bc_bond_A * bc_dir

    # Build sidechain (e.g. =O): on the opposite side of the plane from C.
    # Angle A–B=O (sidechain_ab_angle_deg) measured at B between B→A and B→sidechain.
    sidechain_pos = None
    if geom.sidechain_bond_A > 0.01:
        angle_side = np.deg2rad(geom.sidechain_ab_angle_deg)
        side_dir = (
            ba_dir * np.cos(angle_side)
            - perp_bc * np.sin(angle_side)
        )
        side_dir = _normalize(side_dir)
        sidechain_pos = b_pos + geom.sidechain_bond_A * side_dir

    return b_pos, c_pos, sidechain_pos
