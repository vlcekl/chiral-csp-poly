# poly_csp/chemistry/functionalization.py
from __future__ import annotations

from collections import deque
import json
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.chemistry.linkage import (
    CARBAMATE,
    ESTER,
    LINKAGE_TABLE,
    LinkageGeometry,
    build_linkage_coords,
)
from poly_csp.chemistry.monomers import GlucoseMonomerTemplate
from poly_csp.chemistry.selectors import SelectorRegistry, SelectorTemplate
from poly_csp.chemistry.utils import (
    copy_mol_props,
    coords_from_mol,
    residue_label_maps,
)
from poly_csp.config.schema import SelectorPoseSpec, Site
from poly_csp.geometry.dihedrals import set_dihedral_rad
from poly_csp.geometry.local_frames import compute_residue_local_frame, pose_selector_in_frame


def residue_atom_global_index(
    residue_index: int,
    monomer_atom_count: int,
    local_atom_index: int,
) -> int:
    """Map residue-local atom index -> polymer-global atom index."""
    if residue_index < 0:
        raise ValueError(f"residue_index must be >= 0, got {residue_index}")
    if monomer_atom_count <= 0:
        raise ValueError(f"monomer_atom_count must be > 0, got {monomer_atom_count}")
    if local_atom_index < 0 or local_atom_index >= monomer_atom_count:
        raise ValueError(f"local_atom_index out of range: {local_atom_index}")
    return residue_index * monomer_atom_count + local_atom_index


def _site_to_oxygen_label(site: Site) -> str:
    return f"O{site[1:]}"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Degenerate vector for selector placement.")
    return v / n


def _annotate_selector_atoms(
    rw: Chem.RWMol,
    selector: SelectorTemplate,
    offset: int,
    residue_index: int,
    site: Site,
    instance_id: int,
) -> None:
    for local_idx in range(selector.mol.GetNumAtoms()):
        atom = rw.GetAtomWithIdx(offset + local_idx)
        atom.SetIntProp("_poly_csp_selector_instance", instance_id)
        atom.SetIntProp("_poly_csp_residue_index", residue_index)
        atom.SetProp("_poly_csp_site", site)
        atom.SetIntProp("_poly_csp_selector_local_idx", local_idx)


def _residue_label_global_index(mol: Chem.Mol, residue_index: int, label: str) -> int:
    maps = residue_label_maps(mol)
    if residue_index < 0 or residue_index >= len(maps):
        raise ValueError(f"residue_index {residue_index} out of range [0, {len(maps)})")
    mapping = maps[residue_index]
    if label not in mapping:
        raise ValueError(f"Label {label!r} is unavailable for residue {residue_index}.")
    return int(mapping[label])


def _place_selector_with_ideal_linkage(
    existing_coords: np.ndarray,
    mol_polymer: Chem.Mol,
    residue_index: int,
    site: Site,
    selector: SelectorTemplate,
    linkage_type: str = "carbamate",
) -> np.ndarray:
    """Place selector coordinates using ideal linkage internal coordinates.

    Builds the bridging fragment (e.g. O–C(=O)–NH for carbamate) from
    ideal bond lengths and angles, then orients the selector aromatic
    system relative to the amide nitrogen using the residue local frame.
    """
    oxygen_label = _site_to_oxygen_label(site)
    carbon_label = str(site)  # e.g. "C6"

    # Anchor positions from the polymer
    o_idx = _residue_label_global_index(mol_polymer, residue_index, oxygen_label)
    c_idx = _residue_label_global_index(mol_polymer, residue_index, carbon_label)
    p_o = existing_coords[o_idx]  # anchor: sugar oxygen
    p_c = existing_coords[c_idx]  # anchor parent: sugar carbon

    geom = LINKAGE_TABLE.get(linkage_type, CARBAMATE)

    # Compute a plane reference from the residue local frame
    label_map = residue_label_maps(mol_polymer)[residue_index]
    frame_labels = ("C1", "C2", "C3", "C4", "O4")
    coords_res = np.array([existing_coords[label_map[lab]] for lab in frame_labels])
    frame_idx = {lab: i for i, lab in enumerate(frame_labels)}
    r_res, _ = compute_residue_local_frame(coords_res, frame_idx)
    plane_ref = r_res[2]  # z-axis of residue frame as plane reference

    # Build ideal linkage coordinates
    b_pos, c_pos, sidechain_pos = build_linkage_coords(
        anchor_pos=p_o,
        anchor_parent_pos=p_c,
        geom=geom,
        plane_ref=plane_ref,
    )

    # b_pos = carbonyl carbon position (attach_atom of selector after dummy removal)
    # c_pos = amide nitrogen position
    # sidechain_pos = carbonyl =O position

    # Now place the selector relative to the linkage.
    # The selector template has its own conformer; we need to overlay
    # the attach atom onto b_pos and orient the next atom toward c_pos.
    sel_xyz = np.asarray(
        selector.mol.GetConformer(0).GetPositions(), dtype=float
    ).reshape((-1, 3))

    # Center on attach atom
    centered = sel_xyz - sel_xyz[selector.attach_atom_idx]

    # Build a rotation that maps the selector's attach→next_atom direction
    # to the b_pos→c_pos direction.
    # Identify the next atom from the attach atom in the selector.
    attach_atom = selector.mol.GetAtomWithIdx(selector.attach_atom_idx)
    next_atoms = [
        nbr.GetIdx() for nbr in attach_atom.GetNeighbors()
        if nbr.GetIdx() != selector.attach_dummy_idx
    ]
    if not next_atoms:
        # Fallback: use centroid direction
        selector_dir = _normalize(centered.mean(axis=0))
    else:
        # Use the direction to the primary neighbor (e.g. amide N)
        selector_dir = _normalize(centered[next_atoms[0]])

    target_dir = _normalize(c_pos - b_pos)

    # Rodrigues rotation from selector_dir to target_dir
    rot = _rotation_between_vectors(selector_dir, target_dir)

    # If there's a sidechain (=O), use it to fix the roll rotation.
    if sidechain_pos is not None:
        rotated = centered @ rot.T
        # Find the carbonyl O in the selector (if identifiable)
        carbonyl_o_candidates = [
            nbr.GetIdx() for nbr in attach_atom.GetNeighbors()
            if nbr.GetIdx() != selector.attach_dummy_idx
            and selector.mol.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() == 8
        ]
        if carbonyl_o_candidates:
            co_local = rotated[carbonyl_o_candidates[0]]
            co_target = sidechain_pos - b_pos
            # Project both into the plane perpendicular to target_dir
            co_local_proj = co_local - np.dot(co_local, target_dir) * target_dir
            co_target_proj = co_target - np.dot(co_target, target_dir) * target_dir
            if (
                float(np.linalg.norm(co_local_proj)) > 1e-8
                and float(np.linalg.norm(co_target_proj)) > 1e-8
            ):
                roll = _rotation_between_vectors(
                    _normalize(co_local_proj),
                    _normalize(co_target_proj),
                )
                rotated = rotated @ roll.T
        placed = rotated + b_pos
    else:
        placed = centered @ rot.T + b_pos

    # Remove the dummy atom coordinate if applicable
    if selector.attach_dummy_idx is not None:
        placed = np.delete(placed, selector.attach_dummy_idx, axis=0)

    return placed


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix that maps unit vector *a* to unit vector *b*."""
    a_u = _normalize(a)
    b_u = _normalize(b)
    c = float(np.dot(a_u, b_u))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-12:
        # 180-degree rotation
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(trial, a_u))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _normalize(np.cross(a_u, trial))
        x, y, z = axis
        return np.array([
            [2*x*x - 1, 2*x*y,     2*x*z],
            [2*x*y,     2*y*y - 1, 2*y*z],
            [2*x*z,     2*y*z,     2*z*z - 1],
        ], dtype=float)
    v = np.cross(a_u, b_u)
    s = float(np.linalg.norm(v))
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def attach_selector(
    mol_polymer: Chem.Mol,
    template: GlucoseMonomerTemplate,
    residue_index: int,
    site: Site,
    selector: SelectorTemplate,
    mode: Literal["bond_from_OH_oxygen"] = "bond_from_OH_oxygen",
    linkage_type: str = "carbamate",
) -> Chem.Mol:
    """Chemically attach selector at a site with ideal linkage geometry.

    Uses explicit carbamate (or ester/ether) internal coordinates for the
    bridging fragment, then orients the selector aromatic system using
    the residue local frame.
    """
    if mode != "bond_from_OH_oxygen":
        raise ValueError(f"Unsupported attachment mode: {mode!r}")

    dp = (
        int(mol_polymer.GetIntProp("_poly_csp_dp"))
        if mol_polymer.HasProp("_poly_csp_dp")
        else (mol_polymer.GetNumAtoms() // template.mol.GetNumAtoms())
    )
    if residue_index < 0 or residue_index >= dp:
        raise ValueError(f"residue_index {residue_index} out of range [0, {dp})")

    oxygen_label = _site_to_oxygen_label(site)
    if oxygen_label not in template.site_idx:
        raise ValueError(f"Site {site} is not available in template.site_idx")

    sugar_o_global = _residue_label_global_index(mol_polymer, residue_index, oxygen_label)

    existing_coords = None
    if mol_polymer.GetNumConformers() > 0:
        existing_coords = np.asarray(
            mol_polymer.GetConformer(0).GetPositions(), dtype=float
        ).reshape((-1, 3))

    rw = Chem.RWMol(Chem.CombineMols(mol_polymer, selector.mol))
    offset = mol_polymer.GetNumAtoms()
    attach_global = offset + selector.attach_atom_idx

    prev_count = (
        int(mol_polymer.GetIntProp("_poly_csp_selector_count"))
        if mol_polymer.HasProp("_poly_csp_selector_count")
        else 0
    )
    instance_id = prev_count + 1
    _annotate_selector_atoms(
        rw=rw,
        selector=selector,
        offset=offset,
        residue_index=residue_index,
        site=site,
        instance_id=instance_id,
    )

    rw.AddBond(sugar_o_global, attach_global, Chem.BondType.SINGLE)

    # Compute selector coordinates using ideal linkage geometry
    selector_coords = None
    if existing_coords is not None and selector.mol.GetNumConformers() > 0:
        selector_coords = _place_selector_with_ideal_linkage(
            existing_coords=existing_coords,
            mol_polymer=mol_polymer,
            residue_index=residue_index,
            site=site,
            selector=selector,
            linkage_type=linkage_type,
        )

    if selector.attach_dummy_idx is not None:
        rw.RemoveAtom(offset + selector.attach_dummy_idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    copy_mol_props(mol_polymer, mol)
    mol.SetIntProp("_poly_csp_selector_count", instance_id)

    if existing_coords is not None and selector_coords is not None:
        merged = np.concatenate([existing_coords, selector_coords], axis=0)

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(merged):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

    return mol


def place_selector_coords(
    poly_coords: np.ndarray,
    coords_res: np.ndarray,
    selector_coords: np.ndarray,
    pose: SelectorPoseSpec,
) -> np.ndarray:
    """Rigidly place selector coordinates using residue-local frame + pose rules."""
    # Default labels for the current monomer template atom ordering.
    default_labels = {"C1": 0, "C2": 1, "C3": 3, "C4": 5, "O4": 6}
    r, t = compute_residue_local_frame(coords_res, default_labels)
    return pose_selector_in_frame(
        selector_coords=selector_coords,
        pose=pose,
        r_res=r,
        t_res=t,
        attach_atom_idx=0,
    )


def merge_conformers(poly_coords: np.ndarray, selector_coords: np.ndarray) -> np.ndarray:
    """Concatenate coordinate arrays; mapping handled at attachment time."""
    p = np.asarray(poly_coords, dtype=float).reshape((-1, 3))
    s = np.asarray(selector_coords, dtype=float).reshape((-1, 3))
    return np.concatenate([p, s], axis=0)


def _site_oxygen_global_index(mol: Chem.Mol, residue_index: int, site: Site) -> int:
    o_label = _site_to_oxygen_label(site)
    if mol.HasProp("_poly_csp_residue_label_map_json"):
        return _residue_label_global_index(mol, residue_index, o_label)
    if not mol.HasProp("_poly_csp_template_atom_count"):
        raise ValueError("Missing _poly_csp_template_atom_count metadata on molecule.")
    n_monomer = int(mol.GetIntProp("_poly_csp_template_atom_count"))
    prop = f"_poly_csp_siteidx_{o_label}"
    if not mol.HasProp(prop):
        raise ValueError(f"Missing {prop} metadata on molecule.")
    local_o = int(mol.GetIntProp(prop))
    return residue_atom_global_index(residue_index, n_monomer, local_o)


def _selector_local_to_global_map(
    mol: Chem.Mol,
    residue_index: int,
    site: Site,
) -> dict[int, int]:
    instances: dict[int, dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if int(atom.GetIntProp("_poly_csp_residue_index")) != residue_index:
            continue
        if atom.GetProp("_poly_csp_site") != site:
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        instances.setdefault(inst, {})[local] = atom.GetIdx()

    if not instances:
        raise ValueError(
            f"No selector atoms found for residue {residue_index}, site {site}."
        )
    selected_inst = max(instances.keys())
    return instances[selected_inst]


def _downstream_mask(mol: Chem.Mol, b: int, c: int) -> np.ndarray:
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    mask = np.zeros((n,), dtype=bool)
    q: deque[int] = deque([c])
    mask[c] = True
    while q:
        x = q.popleft()
        for nbr in adj[x]:
            if x == c and nbr == b:
                continue
            if not mask[nbr]:
                mask[nbr] = True
                q.append(nbr)
    return mask


def apply_selector_pose_dihedrals(
    mol: Chem.Mol,
    residue_index: int,
    site: Site,
    pose_spec: SelectorPoseSpec,
    selector: SelectorTemplate | None = None,
) -> Chem.Mol:
    """
    Apply target selector dihedrals (in degrees) for one attached selector.
    If multiple selectors exist at residue/site, the latest attached instance is used.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before applying dihedrals.")
    if not pose_spec.dihedral_targets_deg:
        return Chem.Mol(mol)

    tpl = selector if selector is not None else SelectorRegistry.get("35dmpc")
    local_to_global = _selector_local_to_global_map(mol, residue_index, site)
    sugar_o_global = _site_oxygen_global_index(mol, residue_index, site)

    conf = mol.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))

    for name, target_deg in pose_spec.dihedral_targets_deg.items():
        if name not in tpl.dihedrals:
            raise KeyError(f"Unknown selector dihedral {name!r} for {tpl.name}.")

        mapped = []
        for local_idx in tpl.dihedrals[name]:
            if local_idx in local_to_global:
                mapped.append(local_to_global[local_idx])
            elif tpl.attach_dummy_idx is not None and local_idx == tpl.attach_dummy_idx:
                mapped.append(sugar_o_global)
            else:
                raise ValueError(
                    f"Could not map selector local index {local_idx} for dihedral {name!r}."
                )

        a, b, c, d = mapped
        rotate_mask = _downstream_mask(mol, b, c)
        coords = set_dihedral_rad(
            coords=coords,
            a=a,
            b=b,
            c=c,
            d=d,
            target_angle_rad=np.deg2rad(float(target_deg)),
            rotate_mask=rotate_mask,
        )

    out = Chem.Mol(mol)
    new_conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        new_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(new_conf, assignId=True)
    return out
