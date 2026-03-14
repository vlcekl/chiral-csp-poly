from __future__ import annotations

from collections import deque

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.config.schema import SelectorPoseSpec, Site
from poly_csp.topology.linkage import CARBAMATE, LINKAGE_TABLE, build_linkage_coords
from poly_csp.topology.reactions import residue_label_global_index, site_to_oxygen_label
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.topology.utils import residue_label_maps
from poly_csp.structure.dihedrals import set_dihedral_rad
from poly_csp.structure.local_frames import compute_residue_local_frame, pose_selector_in_frame


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Degenerate vector for selector placement.")
    return v / n


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix that maps unit vector *a* to unit vector *b*."""
    a_u = _normalize(a)
    b_u = _normalize(b)
    c = float(np.dot(a_u, b_u))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-12:
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


def place_selector_with_ideal_linkage(
    existing_coords: np.ndarray,
    mol_polymer: Chem.Mol,
    residue_index: int,
    site: Site,
    selector: SelectorTemplate,
    linkage_type: str = "carbamate",
) -> np.ndarray:
    """Place selector coordinates from ideal linkage internal coordinates."""
    oxygen_label = site_to_oxygen_label(site)
    carbon_label = str(site)

    o_idx = residue_label_global_index(mol_polymer, residue_index, oxygen_label)
    c_idx = residue_label_global_index(mol_polymer, residue_index, carbon_label)
    p_o = existing_coords[o_idx]
    p_c = existing_coords[c_idx]

    geom = LINKAGE_TABLE.get(linkage_type, CARBAMATE)

    label_map = residue_label_maps(mol_polymer)[residue_index]
    frame_labels = ("C1", "C2", "C3", "C4", "O4")
    coords_res = np.array([existing_coords[label_map[lab]] for lab in frame_labels])
    frame_idx = {lab: i for i, lab in enumerate(frame_labels)}
    r_res, _ = compute_residue_local_frame(coords_res, frame_idx)
    plane_ref = r_res[2]

    b_pos, c_pos, sidechain_pos = build_linkage_coords(
        anchor_pos=p_o,
        anchor_parent_pos=p_c,
        geom=geom,
        plane_ref=plane_ref,
    )

    sel_xyz = np.asarray(
        selector.mol.GetConformer(0).GetPositions(), dtype=float
    ).reshape((-1, 3))
    centered = sel_xyz - sel_xyz[selector.attach_atom_idx]

    attach_atom = selector.mol.GetAtomWithIdx(selector.attach_atom_idx)
    next_atoms = [
        nbr.GetIdx() for nbr in attach_atom.GetNeighbors()
        if nbr.GetIdx() != selector.attach_dummy_idx
    ]
    if not next_atoms:
        selector_dir = _normalize(centered.mean(axis=0))
    else:
        selector_dir = _normalize(centered[next_atoms[0]])

    target_dir = _normalize(c_pos - b_pos)
    rot = _rotation_between_vectors(selector_dir, target_dir)

    if sidechain_pos is not None:
        rotated = centered @ rot.T
        carbonyl_o_candidates = [
            nbr.GetIdx() for nbr in attach_atom.GetNeighbors()
            if nbr.GetIdx() != selector.attach_dummy_idx
            and selector.mol.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() == 8
        ]
        if carbonyl_o_candidates:
            co_local = rotated[carbonyl_o_candidates[0]]
            co_target = sidechain_pos - b_pos
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

    if selector.attach_dummy_idx is not None:
        placed = np.delete(placed, selector.attach_dummy_idx, axis=0)

    return placed


def place_selector_coords(
    poly_coords: np.ndarray,
    coords_res: np.ndarray,
    selector_coords: np.ndarray,
    pose: SelectorPoseSpec,
) -> np.ndarray:
    """Rigidly place selector coordinates using residue-local frame + pose rules."""
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
    p = np.asarray(poly_coords, dtype=float).reshape((-1, 3))
    s = np.asarray(selector_coords, dtype=float).reshape((-1, 3))
    return np.concatenate([p, s], axis=0)


def _site_oxygen_global_index(mol: Chem.Mol, residue_index: int, site: Site) -> int:
    o_label = site_to_oxygen_label(site)
    if mol.HasProp("_poly_csp_residue_label_map_json"):
        return residue_label_global_index(mol, residue_index, o_label)
    if not mol.HasProp("_poly_csp_template_atom_count"):
        raise ValueError("Missing _poly_csp_template_atom_count metadata on molecule.")
    n_monomer = int(mol.GetIntProp("_poly_csp_template_atom_count"))
    prop = f"_poly_csp_siteidx_{o_label}"
    if not mol.HasProp(prop):
        raise ValueError(f"Missing {prop} metadata on molecule.")
    local_o = int(mol.GetIntProp(prop))
    if residue_index < 0:
        raise ValueError(f"residue_index must be >= 0, got {residue_index}")
    return residue_index * n_monomer + local_o


def _site_carbon_global_index(mol: Chem.Mol, residue_index: int, site: Site) -> int:
    return residue_label_global_index(mol, residue_index, str(site))


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


def _selector_dihedral_refs(
    selector: SelectorTemplate,
    name: str,
) -> tuple[object, object, object, object]:
    if name in selector.dihedrals:
        return selector.dihedrals[name]
    if name in selector.anchor_dihedrals:
        return selector.anchor_dihedrals[name]
    raise KeyError(f"Unknown selector dihedral {name!r} for {selector.name}.")


def _resolve_selector_dihedral_atom_indices(
    mol: Chem.Mol,
    residue_index: int,
    site: Site,
    selector: SelectorTemplate,
    name: str,
) -> tuple[int, int, int, int]:
    local_to_global = _selector_local_to_global_map(mol, residue_index, site)
    sugar_o_global = _site_oxygen_global_index(mol, residue_index, site)
    site_c_global = _site_carbon_global_index(mol, residue_index, site)

    mapped: list[int] = []
    for ref in _selector_dihedral_refs(selector, name):
        if isinstance(ref, str):
            if ref == "site_oxygen":
                mapped.append(int(sugar_o_global))
                continue
            if ref == "site_carbon":
                mapped.append(int(site_c_global))
                continue
            raise ValueError(
                f"Unsupported selector dihedral atom ref {ref!r} for dihedral {name!r}."
            )
        local_idx = int(ref)
        if local_idx in local_to_global:
            mapped.append(local_to_global[local_idx])
            continue
        if selector.attach_dummy_idx is not None and local_idx == selector.attach_dummy_idx:
            mapped.append(int(sugar_o_global))
            continue
        raise ValueError(
            f"Could not map selector local index {local_idx} for dihedral {name!r}."
        )
    return int(mapped[0]), int(mapped[1]), int(mapped[2]), int(mapped[3])


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
    selector: SelectorTemplate,
) -> Chem.Mol:
    """Apply target selector dihedrals (degrees) for one attached selector."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before applying dihedrals.")
    if not pose_spec.dihedral_targets_deg:
        return Chem.Mol(mol)

    conf = mol.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))

    for name, target_deg in pose_spec.dihedral_targets_deg.items():
        a, b, c, d = _resolve_selector_dihedral_atom_indices(
            mol,
            residue_index,
            site,
            selector,
            name,
        )
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
