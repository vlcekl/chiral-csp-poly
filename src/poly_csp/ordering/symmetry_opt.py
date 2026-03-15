from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
from openmm import unit
from rdkit import Chem
from scipy.optimize import differential_evolution

from poly_csp.config.schema import Site
from poly_csp.forcefield.minimization import (
    new_context,
    potential_energy_kj_mol,
    update_rdkit_coords,
)
from poly_csp.forcefield.runtime_params import RuntimeParams, load_runtime_params
from poly_csp.ordering.hbonds import HbondMetrics, resolve_hbond_connectivity_policy
from poly_csp.ordering.optimize import (
    OrderingSpec,
    _ordering_hbond_diagnostics,
    _ordering_diagnostics,
    _ordering_objective_label,
    _prepare_runtime_ordering_systems,
    _require_forcefield_molecule,
)
from poly_csp.ordering.rotamers import (
    RotamerGridSpec,
    default_rotamer_grid,
)
from poly_csp.ordering.scoring import selector_screw_symmetry_rmsd_from_mol
from poly_csp.structure.alignment import (
    _downstream_mask,
    _resolve_selector_dihedral_atom_indices,
)
from poly_csp.structure.dihedrals import measure_dihedral_rad, set_dihedral_rad
from poly_csp.structure.matrix import ScrewTransform
from poly_csp.topology.reactions import residue_label_global_index
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class _SiteBlock:
    site: str
    local_indices: tuple[int, ...]
    reference_indices: tuple[int, ...]
    residue_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _ActiveDihedral:
    site: str
    name: str
    atom_indices: tuple[int, int, int, int]
    rotate_mask: np.ndarray
    grid_values_deg: tuple[float, ...]


@dataclass(frozen=True)
class _CoupledDihedralInstance:
    atom_indices: tuple[int, int, int, int]
    rotate_mask: np.ndarray


@dataclass(frozen=True)
class _CoupledDihedral:
    kind: str
    name: str
    site: str | None
    instances: tuple[_CoupledDihedralInstance, ...]
    grid_values_deg: tuple[float, ...]


@dataclass(frozen=True)
class _NetworkObjectiveStage:
    label: str
    min_heavy_distance_A: float
    clash_penalty: float
    weight_geom_occ: float
    weight_like_occ: float
    weight_geom_frac: float
    weight_family_min_geom: float
    weight_family_min_like: float
    weight_soft_energy: float
    weight_full_energy: float
    energy_clip: float
    use_full_energy_in_search: bool


@dataclass
class _SymmetryOrderingEngine:
    mol: Chem.Mol
    selector: SelectorTemplate
    spec: OrderingSpec
    sites: tuple[str, ...]
    active_dihedrals: tuple[_ActiveDihedral, ...]
    site_blocks: tuple[_SiteBlock, ...]
    screw: ScrewTransform
    base_coords_A: np.ndarray
    soft_context: object
    soft_integrator: object
    full_context: object | None
    full_integrator: object | None
    soft_nonbonded_mode: str
    full_nonbonded_mode: str | None
    soft_exception_summary: dict[str, object]

    def initial_dihedrals_deg(self) -> dict[str, dict[str, float]]:
        return self.measure_dihedrals_deg(self.base_coords_A)

    def measure_dihedrals_deg(self, coords_A: np.ndarray) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {site: {} for site in self.sites}
        for term in self.active_dihedrals:
            angle_deg = float(
                np.rad2deg(measure_dihedral_rad(coords_A, *term.atom_indices))
            )
            out[term.site][term.name] = angle_deg
        return out

    def build_coords(self, values_deg: np.ndarray) -> np.ndarray:
        coords = np.asarray(self.base_coords_A, dtype=float).copy()
        for value_deg, term in zip(values_deg, self.active_dihedrals, strict=True):
            coords = set_dihedral_rad(
                coords=coords,
                a=term.atom_indices[0],
                b=term.atom_indices[1],
                c=term.atom_indices[2],
                d=term.atom_indices[3],
                target_angle_rad=np.deg2rad(float(value_deg)),
                rotate_mask=term.rotate_mask,
            )
        self._project_from_reference(coords)
        return coords

    def soft_energy_kj_mol(self, coords_A: np.ndarray) -> float:
        return self._energy_kj_mol(self.soft_context, coords_A)

    def full_energy_kj_mol(self, coords_A: np.ndarray) -> float:
        if self.full_context is None:
            return self.soft_energy_kj_mol(coords_A)
        return self._energy_kj_mol(self.full_context, coords_A)

    def _project_from_reference(self, coords_A: np.ndarray) -> None:
        for block in self.site_blocks:
            reference = coords_A[np.asarray(block.reference_indices, dtype=int)]
            for residue_index, target_indices in enumerate(block.residue_indices[1:], start=1):
                coords_A[np.asarray(target_indices, dtype=int)] = self.screw.apply(
                    reference,
                    residue_index,
                )

    @staticmethod
    def _energy_kj_mol(context, coords_A: np.ndarray) -> float:
        positions_nm = (np.asarray(coords_A, dtype=float) / 10.0) * unit.nanometer
        context.setPositions(positions_nm)
        return float(potential_energy_kj_mol(context))


@dataclass
class _SymmetryBackboneRefinementEngine:
    mol: Chem.Mol
    selector: SelectorTemplate
    spec: OrderingSpec
    sites: tuple[str, ...]
    active_terms: tuple[_CoupledDihedral, ...]
    base_coords_A: np.ndarray
    soft_context: object
    soft_integrator: object
    full_context: object | None
    full_integrator: object | None
    soft_nonbonded_mode: str
    full_nonbonded_mode: str | None
    soft_exception_summary: dict[str, object]

    @staticmethod
    def _wrap_angle_deg(theta_deg: float) -> float:
        return float((float(theta_deg) + 180.0) % 360.0 - 180.0)

    def _residue_atom_name_index(self, residue_index: int, atom_name: str) -> int | None:
        matches = [
            int(atom.GetIdx())
            for atom in self.mol.GetAtoms()
            if atom.HasProp("_poly_csp_residue_index")
            and int(atom.GetIntProp("_poly_csp_residue_index")) == int(residue_index)
            and atom.HasProp("_poly_csp_atom_name")
            and atom.GetProp("_poly_csp_atom_name") == str(atom_name)
        ]
        if len(matches) != 1:
            return None
        return int(matches[0])

    def _literature_backbone_term_deg(
        self,
        coords_A: np.ndarray,
        term: _CoupledDihedral,
    ) -> float | None:
        if not term.instances or term.kind != "backbone":
            return None
        atom_indices = term.instances[0].atom_indices
        if term.name == "bb_phi":
            c1_idx = int(atom_indices[1])
            o4_idx = int(atom_indices[2])
            c4_idx = int(atom_indices[3])
            residue_index = int(self.mol.GetAtomWithIdx(c1_idx).GetIntProp("_poly_csp_residue_index"))
            h1_idx = self._residue_atom_name_index(residue_index, "H1")
            if h1_idx is None:
                return None
            return float(
                np.rad2deg(
                    measure_dihedral_rad(
                        coords_A,
                        h1_idx,
                        c1_idx,
                        o4_idx,
                        c4_idx,
                    )
                )
            )
        if term.name == "bb_psi":
            c4_idx = int(atom_indices[1])
            o4_idx = int(atom_indices[2])
            c1_idx = int(atom_indices[3])
            residue_index = int(self.mol.GetAtomWithIdx(c4_idx).GetIntProp("_poly_csp_residue_index"))
            h4_idx = self._residue_atom_name_index(residue_index, "H4")
            if h4_idx is None:
                return None
            return float(
                np.rad2deg(
                    measure_dihedral_rad(
                        coords_A,
                        c1_idx,
                        o4_idx,
                        c4_idx,
                        h4_idx,
                    )
                )
            )
        return None

    def _initial_term_value_deg(self, term: _CoupledDihedral) -> float:
        if term.kind == "backbone":
            prop_name = {
                "bb_phi": "_poly_csp_helix_glycosidic_phi_deg",
                "bb_psi": "_poly_csp_helix_glycosidic_psi_deg",
                "bb_c6_omega": "_poly_csp_helix_glycosidic_omega_deg",
            }.get(term.name)
            if prop_name and self.mol.HasProp(prop_name):
                target_value_deg = float(self.mol.GetDoubleProp(prop_name))
                if term.name in {"bb_phi", "bb_psi"}:
                    literature_value_deg = self._literature_backbone_term_deg(
                        self.base_coords_A,
                        term,
                    )
                    internal_value_deg = self.measure_term_deg(self.base_coords_A, term)
                    if literature_value_deg is not None:
                        delta_deg = self._wrap_angle_deg(
                            target_value_deg - literature_value_deg
                        )
                        return self._wrap_angle_deg(internal_value_deg + delta_deg)
                return float(target_value_deg)
        return self.measure_term_deg(self.base_coords_A, term)

    def initial_values_deg(self) -> np.ndarray:
        return np.asarray(
            [self._initial_term_value_deg(term) for term in self.active_terms],
            dtype=float,
        )

    def selector_dihedrals_deg(self, coords_A: np.ndarray) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {site: {} for site in self.sites}
        for term in self.active_terms:
            if term.kind != "selector" or term.site is None:
                continue
            out.setdefault(term.site, {})[term.name] = self.measure_term_deg(coords_A, term)
        return out

    def backbone_dihedrals_deg(self, coords_A: np.ndarray) -> dict[str, float]:
        return {
            term.name: self.measure_term_deg(coords_A, term)
            for term in self.active_terms
            if term.kind == "backbone"
        }

    def measure_term_deg(self, coords_A: np.ndarray, term: _CoupledDihedral) -> float:
        if not term.instances:
            return 0.0
        return float(
            np.rad2deg(
                measure_dihedral_rad(coords_A, *term.instances[0].atom_indices)
            )
        )

    def build_coords(self, values_deg: np.ndarray) -> np.ndarray:
        coords = np.asarray(self.base_coords_A, dtype=float).copy()
        for value_deg, term in zip(values_deg, self.active_terms, strict=True):
            target_angle_rad = np.deg2rad(float(value_deg))
            for instance in term.instances:
                coords = set_dihedral_rad(
                    coords=coords,
                    a=instance.atom_indices[0],
                    b=instance.atom_indices[1],
                    c=instance.atom_indices[2],
                    d=instance.atom_indices[3],
                    target_angle_rad=target_angle_rad,
                    rotate_mask=instance.rotate_mask,
                )
        return coords

    def soft_energy_kj_mol(self, coords_A: np.ndarray) -> float:
        return _SymmetryOrderingEngine._energy_kj_mol(self.soft_context, coords_A)

    def full_energy_kj_mol(self, coords_A: np.ndarray) -> float:
        if self.full_context is None:
            return self.soft_energy_kj_mol(coords_A)
        return _SymmetryOrderingEngine._energy_kj_mol(self.full_context, coords_A)


def _resolved_screw_transform(mol: Chem.Mol) -> ScrewTransform:
    if not mol.HasProp("_poly_csp_helix_theta_rad") or not mol.HasProp("_poly_csp_helix_rise_A"):
        raise ValueError(
            "symmetry ordering requires _poly_csp_helix_theta_rad and "
            "_poly_csp_helix_rise_A metadata on the forcefield-domain molecule."
        )
    return ScrewTransform(
        theta_rad=float(mol.GetDoubleProp("_poly_csp_helix_theta_rad")),
        rise_A=float(mol.GetDoubleProp("_poly_csp_helix_rise_A")),
    )


def _is_periodic_end_mode(mol: Chem.Mol) -> bool:
    return mol.HasProp("_poly_csp_end_mode") and mol.GetProp("_poly_csp_end_mode") == "periodic"


def _residue_forward_mask(
    mol: Chem.Mol,
    *,
    start_residue_index: int,
) -> np.ndarray:
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_residue_index"):
            continue
        mask[int(atom.GetIdx())] = (
            int(atom.GetIntProp("_poly_csp_residue_index")) >= int(start_residue_index)
        )
    return mask


def _residue_prefix_mask(
    mol: Chem.Mol,
    *,
    end_residue_index: int,
) -> np.ndarray:
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_residue_index"):
            continue
        mask[int(atom.GetIdx())] = (
            int(atom.GetIntProp("_poly_csp_residue_index")) <= int(end_residue_index)
        )
    return mask


def _atom_block_signature(atom: Chem.Atom) -> tuple[str, str, str, str, int]:
    return (
        atom.GetProp("_poly_csp_manifest_source")
        if atom.HasProp("_poly_csp_manifest_source")
        else "",
        atom.GetProp("_poly_csp_atom_name") if atom.HasProp("_poly_csp_atom_name") else "",
        atom.GetProp("_poly_csp_site") if atom.HasProp("_poly_csp_site") else "",
        (
            atom.GetProp("_poly_csp_connector_role")
            if atom.HasProp("_poly_csp_connector_role")
            else ""
        ),
        (
            int(atom.GetIntProp("_poly_csp_selector_local_idx"))
            if atom.HasProp("_poly_csp_selector_local_idx")
            else -1
        ),
    )


def _residue_blocks(
    mol: Chem.Mol,
    *,
    dp: int,
) -> tuple[tuple[int, ...], ...] | None:
    blocks: list[tuple[int, ...]] = []
    reference_signature: tuple[tuple[str, str, str, str, int], ...] | None = None
    for residue_index in range(int(dp)):
        block = tuple(
            int(atom.GetIdx())
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_residue_index")
            and int(atom.GetIntProp("_poly_csp_residue_index")) == int(residue_index)
        )
        if not block:
            return None
        signature = tuple(
            _atom_block_signature(mol.GetAtomWithIdx(atom_idx))
            for atom_idx in block
        )
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            return None
        blocks.append(block)
    return tuple(blocks)


def _lifted_residue_mask(
    lifted_residue_indices: np.ndarray,
    *,
    lower: int | None = None,
    upper: int | None = None,
) -> np.ndarray:
    mask = np.ones_like(lifted_residue_indices, dtype=bool)
    if lower is not None:
        mask &= lifted_residue_indices >= int(lower)
    if upper is not None:
        mask &= lifted_residue_indices <= int(upper)
    return np.asarray(mask, dtype=bool)


def _site_attachment_local_to_global_map(
    mol: Chem.Mol,
    *,
    residue_index: int,
    site: str,
) -> dict[int, int]:
    instances: dict[int, dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        if not atom.HasProp("_poly_csp_residue_index") or not atom.HasProp("_poly_csp_site"):
            continue
        if int(atom.GetIntProp("_poly_csp_residue_index")) != int(residue_index):
            continue
        if atom.GetProp("_poly_csp_site") != str(site):
            continue
        instance = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local_idx = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        instances.setdefault(instance, {})[local_idx] = int(atom.GetIdx())
    if not instances:
        raise ValueError(
            f"No attached selector block found for residue {residue_index}, site {site!r}."
        )
    selected_instance = max(instances)
    return dict(instances[selected_instance])


def _site_block(
    mol: Chem.Mol,
    *,
    dp: int,
    site: str,
) -> _SiteBlock:
    reference = _site_attachment_local_to_global_map(mol, residue_index=0, site=site)
    local_indices = tuple(sorted(reference))
    residue_indices: list[tuple[int, ...]] = []
    for residue_index in range(int(dp)):
        mapping = _site_attachment_local_to_global_map(
            mol,
            residue_index=residue_index,
            site=site,
        )
        if tuple(sorted(mapping)) != local_indices:
            raise ValueError(
                f"Selector local-index mismatch for site {site!r} at residue {residue_index}."
            )
        residue_indices.append(tuple(mapping[local_idx] for local_idx in local_indices))
    return _SiteBlock(
        site=str(site),
        local_indices=local_indices,
        reference_indices=tuple(reference[local_idx] for local_idx in local_indices),
        residue_indices=tuple(residue_indices),
    )


def _active_dihedrals(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    sites: Iterable[str],
    active_values_by_name: Mapping[str, tuple[float, ...]],
) -> tuple[_ActiveDihedral, ...]:
    active: list[_ActiveDihedral] = []
    for site in sites:
        for name, grid_values in active_values_by_name.items():
            a, b, c, d = _resolve_selector_dihedral_atom_indices(
                mol,
                0,
                site,  # type: ignore[arg-type]
                selector,
                name,
            )
            active.append(
                _ActiveDihedral(
                    site=str(site),
                    name=str(name),
                    atom_indices=(int(a), int(b), int(c), int(d)),
                    rotate_mask=_downstream_mask(mol, int(b), int(c)),
                    grid_values_deg=tuple(float(value) for value in grid_values),
                )
            )
    return tuple(active)


def _active_grid_values(
    selector: SelectorTemplate,
    *,
    grid: RotamerGridSpec | None,
    include_anchor_dihedrals: bool,
) -> dict[str, tuple[float, ...]]:
    grid_spec = default_rotamer_grid(selector) if grid is None else grid
    values: dict[str, tuple[float, ...]] = {
        str(name): tuple(float(value) for value in dihedral_values)
        for name, dihedral_values in grid_spec.dihedral_values_deg.items()
    }
    if include_anchor_dihedrals:
        for name, dihedral_values in selector.anchor_rotamer_grid.items():
            values.setdefault(
                str(name),
                tuple(float(value) for value in dihedral_values),
            )
    return values


def _coupled_selector_dihedrals(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    sites: Iterable[str],
    dp: int,
    active_values_by_name: Mapping[str, tuple[float, ...]],
) -> tuple[_CoupledDihedral, ...]:
    coupled: list[_CoupledDihedral] = []
    for site in sites:
        for name, grid_values in active_values_by_name.items():
            instances: list[_CoupledDihedralInstance] = []
            for residue_index in range(int(dp)):
                a, b, c, d = _resolve_selector_dihedral_atom_indices(
                    mol,
                    residue_index,
                    site,  # type: ignore[arg-type]
                    selector,
                    name,
                )
                instances.append(
                    _CoupledDihedralInstance(
                        atom_indices=(int(a), int(b), int(c), int(d)),
                        rotate_mask=_downstream_mask(mol, int(b), int(c)),
                    )
                )
            coupled.append(
                _CoupledDihedral(
                    kind="selector",
                    name=str(name),
                    site=str(site),
                    instances=tuple(instances),
                    grid_values_deg=tuple(float(value) for value in grid_values),
                )
            )
    return tuple(coupled)


def _c6_omega_coupled_dihedral(
    mol: Chem.Mol,
    *,
    dp: int,
) -> _CoupledDihedral | None:
    instances: list[_CoupledDihedralInstance] = []
    for residue_index in range(int(dp)):
        try:
            a = residue_label_global_index(mol, residue_index, "O5")
            b = residue_label_global_index(mol, residue_index, "C5")
            c = residue_label_global_index(mol, residue_index, "C6")
            d = residue_label_global_index(mol, residue_index, "O6")
        except ValueError:
            return None
        instances.append(
            _CoupledDihedralInstance(
                atom_indices=(int(a), int(b), int(c), int(d)),
                rotate_mask=_downstream_mask(mol, int(b), int(c)),
            )
        )
    if not instances:
        return None
    return _CoupledDihedral(
        kind="backbone",
        name="bb_c6_omega",
        site=None,
        instances=tuple(instances),
        grid_values_deg=(),
    )


def _phi_coupled_dihedral(
    mol: Chem.Mol,
    *,
    dp: int,
) -> _CoupledDihedral | None:
    if int(dp) < 2 or _is_periodic_end_mode(mol):
        return None
    instances: list[_CoupledDihedralInstance] = []
    for residue_index in range(int(dp) - 1):
        try:
            a = residue_label_global_index(mol, residue_index, "O5")
            b = residue_label_global_index(mol, residue_index, "C1")
            c = residue_label_global_index(mol, residue_index + 1, "O4")
            d = residue_label_global_index(mol, residue_index + 1, "C4")
        except ValueError:
            return None
        rotate_mask = _residue_forward_mask(
            mol,
            start_residue_index=(residue_index + 1),
        )
        instances.append(
            _CoupledDihedralInstance(
                atom_indices=(int(a), int(b), int(c), int(d)),
                rotate_mask=rotate_mask,
            )
        )
    if not instances:
        return None
    return _CoupledDihedral(
        kind="backbone",
        name="bb_phi",
        site=None,
        instances=tuple(instances),
        grid_values_deg=(),
    )


def _psi_coupled_dihedral(
    mol: Chem.Mol,
    *,
    dp: int,
) -> _CoupledDihedral | None:
    if int(dp) < 2 or _is_periodic_end_mode(mol):
        return None
    instances: list[_CoupledDihedralInstance] = []
    for residue_index in range(int(dp) - 1):
        try:
            a = residue_label_global_index(mol, residue_index + 1, "C3")
            b = residue_label_global_index(mol, residue_index + 1, "C4")
            c = residue_label_global_index(mol, residue_index + 1, "O4")
            d = residue_label_global_index(mol, residue_index, "C1")
        except ValueError:
            return None
        rotate_mask = _residue_prefix_mask(
            mol,
            end_residue_index=residue_index,
        )
        instances.append(
            _CoupledDihedralInstance(
                atom_indices=(int(a), int(b), int(c), int(d)),
                rotate_mask=rotate_mask,
            )
        )
    if not instances:
        return None
    return _CoupledDihedral(
        kind="backbone",
        name="bb_psi",
        site=None,
        instances=tuple(instances),
        grid_values_deg=(),
    )


def _periodic_phi_measurement_dihedral(
    mol: Chem.Mol,
    *,
    dp: int,
) -> _CoupledDihedral | None:
    if int(dp) < 2 or not _is_periodic_end_mode(mol):
        return None
    try:
        a = residue_label_global_index(mol, 0, "O5")
        b = residue_label_global_index(mol, 0, "C1")
        c = residue_label_global_index(mol, 1, "O4")
        d = residue_label_global_index(mol, 1, "C4")
    except ValueError:
        return None
    return _CoupledDihedral(
        kind="backbone",
        name="bb_phi",
        site=None,
        instances=(
            _CoupledDihedralInstance(
                atom_indices=(int(a), int(b), int(c), int(d)),
                rotate_mask=_residue_forward_mask(mol, start_residue_index=1),
            ),
        ),
        grid_values_deg=(),
    )


def _periodic_psi_measurement_dihedral(
    mol: Chem.Mol,
    *,
    dp: int,
) -> _CoupledDihedral | None:
    if int(dp) < 2 or not _is_periodic_end_mode(mol):
        return None
    try:
        a = residue_label_global_index(mol, 1, "C3")
        b = residue_label_global_index(mol, 1, "C4")
        c = residue_label_global_index(mol, 1, "O4")
        d = residue_label_global_index(mol, 0, "C1")
    except ValueError:
        return None
    return _CoupledDihedral(
        kind="backbone",
        name="bb_psi",
        site=None,
        instances=(
            _CoupledDihedralInstance(
                atom_indices=(int(a), int(b), int(c), int(d)),
                rotate_mask=_residue_prefix_mask(mol, end_residue_index=0),
            ),
        ),
        grid_values_deg=(),
    )


def _periodic_lifted_backbone_instances(
    mol: Chem.Mol,
    *,
    residue_blocks: tuple[tuple[int, ...], ...],
) -> dict[str, tuple[_CoupledDihedralInstance, ...]] | None:
    dp = len(residue_blocks)
    if dp < 2:
        return None

    reference_block = residue_blocks[0]
    n_atoms = int(mol.GetNumAtoms())
    duplicate_offset = n_atoms
    duplicate_map = {
        int(base_atom_idx): int(duplicate_offset + local_idx)
        for local_idx, base_atom_idx in enumerate(reference_block)
    }

    lifted_residue_indices = np.full((n_atoms + len(reference_block),), -1, dtype=int)
    for residue_index, block in enumerate(residue_blocks):
        lifted_residue_indices[np.asarray(block, dtype=int)] = int(residue_index)
    lifted_residue_indices[duplicate_offset:] = int(dp)

    phi_instances: list[_CoupledDihedralInstance] = []
    psi_instances: list[_CoupledDihedralInstance] = []
    for residue_index in range(dp):
        next_index = residue_index + 1
        try:
            a_phi = residue_label_global_index(mol, residue_index, "O5")
            b_phi = residue_label_global_index(mol, residue_index, "C1")
            d_psi = residue_label_global_index(mol, residue_index, "C1")
            if next_index < dp:
                c_phi = residue_label_global_index(mol, next_index, "O4")
                d_phi = residue_label_global_index(mol, next_index, "C4")
                a_psi = residue_label_global_index(mol, next_index, "C3")
                b_psi = residue_label_global_index(mol, next_index, "C4")
                c_psi = residue_label_global_index(mol, next_index, "O4")
            else:
                c_phi = duplicate_map[residue_label_global_index(mol, 0, "O4")]
                d_phi = duplicate_map[residue_label_global_index(mol, 0, "C4")]
                a_psi = duplicate_map[residue_label_global_index(mol, 0, "C3")]
                b_psi = duplicate_map[residue_label_global_index(mol, 0, "C4")]
                c_psi = duplicate_map[residue_label_global_index(mol, 0, "O4")]
        except ValueError:
            return None

        phi_instances.append(
            _CoupledDihedralInstance(
                atom_indices=(int(a_phi), int(b_phi), int(c_phi), int(d_phi)),
                rotate_mask=_lifted_residue_mask(
                    lifted_residue_indices,
                    lower=next_index,
                ),
            )
        )
        psi_instances.append(
            _CoupledDihedralInstance(
                atom_indices=(int(a_psi), int(b_psi), int(c_psi), int(d_psi)),
                rotate_mask=_lifted_residue_mask(
                    lifted_residue_indices,
                    upper=residue_index,
                ),
            )
        )

    return {
        "bb_phi": tuple(phi_instances),
        "bb_psi": tuple(psi_instances),
    }


@dataclass
class _SymmetryPeriodicBackboneRefinementEngine(_SymmetryBackboneRefinementEngine):
    screw: ScrewTransform
    residue_blocks: tuple[tuple[int, ...], ...]
    duplicate_block: tuple[int, ...]
    periodic_instances_by_name: dict[str, tuple[_CoupledDihedralInstance, ...]]

    def build_coords(self, values_deg: np.ndarray) -> np.ndarray:
        coords = np.asarray(self.base_coords_A, dtype=float).copy()
        periodic_targets_deg: dict[str, float] = {}
        deferred_terms: list[tuple[float, _CoupledDihedral]] = []
        for value_deg, term in zip(values_deg, self.active_terms, strict=True):
            if term.kind == "backbone" and term.name in self.periodic_instances_by_name:
                periodic_targets_deg[term.name] = float(value_deg)
                continue
            deferred_terms.append((float(value_deg), term))

        if periodic_targets_deg:
            coords = self._apply_periodic_backbone_terms(coords, periodic_targets_deg)

        for value_deg, term in deferred_terms:
            target_angle_rad = np.deg2rad(float(value_deg))
            for instance in term.instances:
                coords = set_dihedral_rad(
                    coords=coords,
                    a=instance.atom_indices[0],
                    b=instance.atom_indices[1],
                    c=instance.atom_indices[2],
                    d=instance.atom_indices[3],
                    target_angle_rad=target_angle_rad,
                    rotate_mask=instance.rotate_mask,
                )
        return coords

    def _apply_periodic_backbone_terms(
        self,
        coords_A: np.ndarray,
        targets_deg: Mapping[str, float],
    ) -> np.ndarray:
        lifted = self._lift_coords(coords_A)
        for name in ("bb_phi", "bb_psi"):
            if name not in targets_deg:
                continue
            target_angle_rad = np.deg2rad(float(targets_deg[name]))
            for instance in self.periodic_instances_by_name[name]:
                lifted = set_dihedral_rad(
                    coords=lifted,
                    a=instance.atom_indices[0],
                    b=instance.atom_indices[1],
                    c=instance.atom_indices[2],
                    d=instance.atom_indices[3],
                    target_angle_rad=target_angle_rad,
                    rotate_mask=instance.rotate_mask,
                )
        return self._collapse_lifted_coords(coords_A, lifted)

    def _lift_coords(self, coords_A: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords_A, dtype=float)
        n_atoms = int(coords.shape[0])
        lifted = np.zeros((n_atoms + len(self.duplicate_block), 3), dtype=float)
        lifted[:n_atoms] = coords
        lifted[np.asarray(self.duplicate_block, dtype=int)] = self.screw.apply(
            coords[np.asarray(self.residue_blocks[0], dtype=int)],
            len(self.residue_blocks),
        )
        return lifted

    def _collapse_lifted_coords(
        self,
        coords_A: np.ndarray,
        lifted_coords_A: np.ndarray,
    ) -> np.ndarray:
        collapsed = np.asarray(coords_A, dtype=float).copy()
        reference_blocks: list[np.ndarray] = []
        for residue_index, block in enumerate(self.residue_blocks):
            reference_blocks.append(
                self.screw.apply(
                    lifted_coords_A[np.asarray(block, dtype=int)],
                    -residue_index,
                )
            )
        reference_blocks.append(
            self.screw.apply(
                lifted_coords_A[np.asarray(self.duplicate_block, dtype=int)],
                -len(self.residue_blocks),
            )
        )
        reference_coords = np.mean(np.stack(reference_blocks, axis=0), axis=0)
        for residue_index, block in enumerate(self.residue_blocks):
            collapsed[np.asarray(block, dtype=int)] = self.screw.apply(
                reference_coords,
                residue_index,
            )
        return collapsed


def _build_backbone_refinement_engine(
    engine: _SymmetryOrderingEngine,
    *,
    selector: SelectorTemplate,
    grid: RotamerGridSpec | None,
    dp: int,
    base_coords_A: np.ndarray,
) -> tuple[
    _SymmetryBackboneRefinementEngine | _SymmetryPeriodicBackboneRefinementEngine | None,
    str | None,
]:
    if not bool(engine.spec.symmetry_backbone_refine_enabled):
        return None, "disabled"

    periodic_mode = _is_periodic_end_mode(engine.mol)
    backbone_terms: list[_CoupledDihedral] = []
    if bool(engine.spec.symmetry_backbone_include_c6_omega) and "C6" in engine.sites:
        term = _c6_omega_coupled_dihedral(engine.mol, dp=dp)
        if term is not None:
            backbone_terms.append(term)
    periodic_instances_by_name: dict[str, tuple[_CoupledDihedralInstance, ...]] = {}
    residue_blocks = _residue_blocks(engine.mol, dp=dp) if periodic_mode else None
    if any(site in {"C2", "C3"} for site in engine.sites):
        if bool(engine.spec.symmetry_backbone_include_phi):
            term = (
                _periodic_phi_measurement_dihedral(engine.mol, dp=dp)
                if periodic_mode
                else _phi_coupled_dihedral(engine.mol, dp=dp)
            )
            if term is not None:
                backbone_terms.append(term)
        if bool(engine.spec.symmetry_backbone_include_psi):
            term = (
                _periodic_psi_measurement_dihedral(engine.mol, dp=dp)
                if periodic_mode
                else _psi_coupled_dihedral(engine.mol, dp=dp)
            )
            if term is not None:
                backbone_terms.append(term)
        if periodic_mode and any(term.name in {"bb_phi", "bb_psi"} for term in backbone_terms):
            if residue_blocks is None:
                return None, "periodic_backbone_block_mismatch"
            periodic_instances = _periodic_lifted_backbone_instances(
                engine.mol,
                residue_blocks=residue_blocks,
            )
            if periodic_instances is None:
                return None, "periodic_backbone_instances_unavailable"
            periodic_instances_by_name = dict(periodic_instances)

    if not backbone_terms:
        if periodic_mode and (
            bool(engine.spec.symmetry_backbone_include_phi)
            or bool(engine.spec.symmetry_backbone_include_psi)
        ):
            return None, "periodic_glycosidic_backbone_unsupported"
        return None, "no_active_backbone_dihedrals"

    selector_terms: list[_CoupledDihedral] = []
    if bool(engine.spec.symmetry_backbone_reoptimize_selectors):
        active_values_by_name = _active_grid_values(
            selector,
            grid=grid,
            include_anchor_dihedrals=True,
        )
        selector_terms.extend(
            _coupled_selector_dihedrals(
                engine.mol,
                selector=selector,
                sites=engine.sites,
                dp=dp,
                active_values_by_name={
                    name: active_values_by_name[name]
                    for name in sorted(active_values_by_name)
                },
            )
        )

    active_terms = tuple(backbone_terms + selector_terms)
    common_kwargs = dict(
        mol=update_rdkit_coords(
            engine.mol,
            (np.asarray(base_coords_A, dtype=float) / 10.0) * unit.nanometer,
        ),
        selector=selector,
        spec=engine.spec,
        sites=engine.sites,
        active_terms=active_terms,
        base_coords_A=np.asarray(base_coords_A, dtype=float).copy(),
        soft_context=engine.soft_context,
        soft_integrator=engine.soft_integrator,
        full_context=engine.full_context,
        full_integrator=engine.full_integrator,
        soft_nonbonded_mode=engine.soft_nonbonded_mode,
        full_nonbonded_mode=engine.full_nonbonded_mode,
        soft_exception_summary=dict(engine.soft_exception_summary),
    )
    if periodic_instances_by_name:
        assert residue_blocks is not None
        duplicate_offset = int(engine.mol.GetNumAtoms())
        duplicate_block = tuple(
            duplicate_offset + local_idx for local_idx in range(len(residue_blocks[0]))
        )
        return _SymmetryPeriodicBackboneRefinementEngine(
            **common_kwargs,
            screw=engine.screw,
            residue_blocks=residue_blocks,
            duplicate_block=duplicate_block,
            periodic_instances_by_name=periodic_instances_by_name,
        ), None
    return _SymmetryBackboneRefinementEngine(**common_kwargs), None


def _symmetry_pose_summary(
    values_by_site: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        site: {"0": {name: float(value) for name, value in sorted(dihedrals.items())}}
        for site, dihedrals in values_by_site.items()
    }


def _build_engine(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate,
    sites: Iterable[str],
    dp: int,
    spec: OrderingSpec,
    grid: RotamerGridSpec | None,
    include_anchor_dihedrals: bool,
    runtime_params: RuntimeParams,
    mixing_rules_cfg: Mapping[str, object] | None,
) -> _SymmetryOrderingEngine:
    screw = _resolved_screw_transform(mol)
    site_list = tuple(str(site) for site in sites)
    site_blocks = tuple(
        _site_block(
            mol,
            dp=dp,
            site=site,
        )
        for site in site_list
    )

    active_values_by_name = _active_grid_values(
        selector,
        grid=grid,
        include_anchor_dihedrals=include_anchor_dihedrals,
    )
    active_dihedrals = _active_dihedrals(
        mol,
        selector=selector,
        sites=site_list,
        active_values_by_name={
            name: active_values_by_name[name]
            for name in sorted(active_values_by_name)
        },
    )

    raw_coords_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    base_coords_A = raw_coords_A.copy()
    for block in site_blocks:
        reference = raw_coords_A[np.asarray(block.reference_indices, dtype=int)]
        for residue_index, target_indices in enumerate(block.residue_indices[1:], start=1):
            base_coords_A[np.asarray(target_indices, dtype=int)] = screw.apply(
                reference,
                residue_index,
            )

    base_mol = update_rdkit_coords(mol, (base_coords_A / 10.0) * unit.nanometer)
    prepared = _prepare_runtime_ordering_systems(
        base_mol,
        selector=selector,
        runtime_params=runtime_params,
        spec=spec,
        mixing_rules_cfg=mixing_rules_cfg,
    )
    soft_context, soft_integrator = new_context(
        prepared.soft.system,
        prepared.soft.positions_nm,
    )
    full_context = None
    full_integrator = None
    if not bool(spec.skip_full_stage):
        full_context, full_integrator = new_context(
            prepared.full.system,
            prepared.full.positions_nm,
        )
    return _SymmetryOrderingEngine(
        mol=base_mol,
        selector=selector,
        spec=spec,
        sites=site_list,
        active_dihedrals=active_dihedrals,
        site_blocks=site_blocks,
        screw=screw,
        base_coords_A=base_coords_A,
        soft_context=soft_context,
        soft_integrator=soft_integrator,
        full_context=full_context,
        full_integrator=full_integrator,
        soft_nonbonded_mode=str(prepared.soft.nonbonded_mode),
        full_nonbonded_mode=(
            None if bool(spec.skip_full_stage) else str(prepared.full.nonbonded_mode)
        ),
        soft_exception_summary=dict(prepared.soft.exception_summary),
    )


def _rebase_engine(
    engine: _SymmetryOrderingEngine,
    *,
    base_coords_A: np.ndarray,
) -> _SymmetryOrderingEngine:
    coords_A = np.asarray(base_coords_A, dtype=float).copy()
    return _SymmetryOrderingEngine(
        mol=update_rdkit_coords(engine.mol, (coords_A / 10.0) * unit.nanometer),
        selector=engine.selector,
        spec=engine.spec,
        sites=engine.sites,
        active_dihedrals=engine.active_dihedrals,
        site_blocks=engine.site_blocks,
        screw=engine.screw,
        base_coords_A=coords_A,
        soft_context=engine.soft_context,
        soft_integrator=engine.soft_integrator,
        full_context=engine.full_context,
        full_integrator=engine.full_integrator,
        soft_nonbonded_mode=engine.soft_nonbonded_mode,
        full_nonbonded_mode=engine.full_nonbonded_mode,
        soft_exception_summary=dict(engine.soft_exception_summary),
    )


def _capture_ordering_spec(spec: OrderingSpec) -> OrderingSpec:
    return replace(
        spec,
        skip_full_stage=True,
        soft_repulsion_k_kj_per_mol_nm2=(
            float(spec.soft_repulsion_k_kj_per_mol_nm2)
            * float(spec.symmetry_network_capture_soft_repulsion_scale)
        ),
    )


def _capture_objective_stage(spec: OrderingSpec) -> _NetworkObjectiveStage:
    return _NetworkObjectiveStage(
        label="network_capture",
        min_heavy_distance_A=float(spec.symmetry_network_capture_min_heavy_distance_A),
        clash_penalty=float(spec.symmetry_network_capture_clash_penalty),
        weight_geom_occ=float(spec.symmetry_network_capture_weight_geom_occ),
        weight_like_occ=float(spec.symmetry_network_capture_weight_like_occ),
        weight_geom_frac=float(spec.symmetry_network_capture_weight_geom_frac),
        weight_family_min_geom=float(spec.symmetry_network_capture_weight_family_min_geom),
        weight_family_min_like=float(spec.symmetry_network_capture_weight_family_min_like),
        weight_soft_energy=float(spec.symmetry_network_capture_weight_soft_energy),
        weight_full_energy=0.0,
        energy_clip=float(spec.symmetry_network_capture_energy_clip),
        use_full_energy_in_search=False,
    )


def _cleanup_objective_stage(spec: OrderingSpec) -> _NetworkObjectiveStage:
    return _NetworkObjectiveStage(
        label="network_cleanup",
        min_heavy_distance_A=float(spec.symmetry_network_min_heavy_distance_A),
        clash_penalty=float(spec.symmetry_network_clash_penalty),
        weight_geom_occ=float(spec.symmetry_network_weight_geom_occ),
        weight_like_occ=float(spec.symmetry_network_weight_like_occ),
        weight_geom_frac=float(spec.symmetry_network_weight_geom_frac),
        weight_family_min_geom=float(spec.symmetry_network_weight_family_min_geom),
        weight_family_min_like=float(spec.symmetry_network_weight_family_min_like),
        weight_soft_energy=float(spec.symmetry_network_weight_soft_energy),
        weight_full_energy=float(spec.symmetry_network_weight_full_energy),
        energy_clip=float(spec.symmetry_network_energy_clip),
        use_full_energy_in_search=bool(spec.symmetry_network_use_full_energy_in_search),
    )


@dataclass(frozen=True)
class _CandidateEval:
    values_deg: tuple[float, ...]
    coords_A: np.ndarray
    soft_energy_kj_mol: float
    full_energy_kj_mol: float
    score: float
    hbond_metrics: HbondMetrics
    hbond_like_fraction: float
    hbond_geometric_fraction: float
    hbond_like_donor_occupancy_fraction: float
    hbond_geometric_donor_occupancy_fraction: float
    hbond_like_satisfied_donors: int
    hbond_geometric_satisfied_donors: int
    hbond_family_metrics: dict[str, HbondMetrics]
    hbond_family_min_like_fraction: float
    hbond_family_min_geometric_fraction: float
    min_heavy_distance_A: float
    class_min_distance_A: dict[str, float | None]
    selector_aromatic_stacking_A: dict[str, object]


def _candidate_cache_key(values_deg: np.ndarray) -> tuple[float, ...]:
    arr = np.asarray(values_deg, dtype=float).reshape((-1,))
    return tuple(float(np.round(value, 6)) for value in arr)


def _relative_energy_term(
    energy_kj_mol: float,
    baseline_kj_mol: float,
    clip: float,
) -> float:
    scale = max(abs(float(baseline_kj_mol)), 1.0e3)
    term = (float(energy_kj_mol) - float(baseline_kj_mol)) / scale
    return float(np.clip(term, -abs(float(clip)), abs(float(clip))))


def _family_metric_min(
    family_metrics: Mapping[str, HbondMetrics],
    *,
    attr: str,
) -> float:
    if not family_metrics:
        return 1.0
    return min(float(getattr(metrics, attr)) for metrics in family_metrics.values())


def _hbond_family_metrics_summary(
    family_metrics: Mapping[str, HbondMetrics],
) -> dict[str, dict[str, float | int]]:
    return {
        str(name): {
            "like_fraction": float(metrics.like_fraction),
            "geometric_fraction": float(metrics.geometric_fraction),
            "donor_count": int(metrics.donor_count),
            "like_satisfied_pairs": int(metrics.like_satisfied_pairs),
            "geometric_satisfied_pairs": int(metrics.geometric_satisfied_pairs),
            "total_pairs": int(metrics.total_pairs),
            "like_donor_occupancy_fraction": float(metrics.like_donor_occupancy_fraction),
            "geometric_donor_occupancy_fraction": float(
                metrics.geometric_donor_occupancy_fraction
            ),
            "mean_like_distance_A": float(metrics.mean_like_distance_A),
            "mean_geometric_distance_A": float(metrics.mean_geometric_distance_A),
        }
        for name, metrics in sorted(family_metrics.items())
    }


def _initial_population_from_rotamers(
    engine: _SymmetryOrderingEngine,
    *,
    initial_values_deg: np.ndarray,
    popsize: int,
    jitter_deg: float,
    seed: int | None,
) -> np.ndarray | None:
    dims = int(initial_values_deg.size)
    if dims <= 0:
        return None

    size = max(5, int(popsize) * dims)
    rng = np.random.default_rng(seed)
    population = np.zeros((size, dims), dtype=float)
    population[0] = np.asarray(initial_values_deg, dtype=float)
    jitter = max(0.0, float(jitter_deg))

    for row_idx in range(1, size):
        for col_idx, term in enumerate(engine.active_dihedrals):
            values = (
                term.grid_values_deg
                if term.grid_values_deg
                else (float(initial_values_deg[col_idx]),)
            )
            base = float(values[int(rng.integers(len(values)))])
            offset = float(rng.uniform(-jitter, jitter)) if jitter > 0.0 else 0.0
            population[row_idx, col_idx] = ((base + offset + 180.0) % 360.0) - 180.0
    return population


def _initial_population_from_coupled_terms(
    active_terms: Iterable[_CoupledDihedral],
    *,
    initial_values_deg: np.ndarray,
    popsize: int,
    jitter_deg: float,
    seed: int | None,
) -> np.ndarray | None:
    initial = np.asarray(initial_values_deg, dtype=float).reshape((-1,))
    dims = int(initial.size)
    if dims <= 0:
        return None

    terms = tuple(active_terms)
    size = max(5, int(popsize) * dims)
    rng = np.random.default_rng(seed)
    population = np.zeros((size, dims), dtype=float)
    population[0] = initial
    jitter = max(0.0, float(jitter_deg))

    for row_idx in range(1, size):
        for col_idx, term in enumerate(terms):
            values = (
                term.grid_values_deg
                if term.grid_values_deg
                else (float(initial[col_idx]),)
            )
            base = float(values[int(rng.integers(len(values)))])
            offset = float(rng.uniform(-jitter, jitter)) if jitter > 0.0 else 0.0
            population[row_idx, col_idx] = ((base + offset + 180.0) % 360.0) - 180.0
    return population


def _evaluate_network_candidate(
    engine: _SymmetryOrderingEngine | _SymmetryBackboneRefinementEngine,
    *,
    selector: SelectorTemplate,
    objective_stage: _NetworkObjectiveStage,
    values_deg: np.ndarray,
    baseline_soft_energy: float,
    baseline_full_energy: float,
    cache: dict[tuple[float, ...], _CandidateEval],
) -> _CandidateEval:
    key = _candidate_cache_key(values_deg)
    cached = cache.get(key)
    if cached is not None:
        return cached

    coords_A = engine.build_coords(np.asarray(values_deg, dtype=float))
    candidate_mol = update_rdkit_coords(engine.mol, (coords_A / 10.0) * unit.nanometer)
    hb_diag = _ordering_hbond_diagnostics(candidate_mol, selector, engine.spec)
    hb = hb_diag.metrics
    family_metrics = dict(hb_diag.family_metrics)
    family_min_like = _family_metric_min(
        family_metrics,
        attr="like_fraction",
    )
    family_min_geom = _family_metric_min(
        family_metrics,
        attr="geometric_fraction",
    )
    _, dmin, class_min, stacking = _ordering_diagnostics(candidate_mol, selector, engine.spec)

    clash_cutoff = float(objective_stage.min_heavy_distance_A)
    clash_penalty = float(objective_stage.clash_penalty)
    clipped_class_min = {
        key_name: (float(value) if np.isfinite(value) else None)
        for key_name, value in class_min.items()
    }
    if float(dmin) < clash_cutoff:
        delta = clash_cutoff - float(dmin)
        result = _CandidateEval(
            values_deg=tuple(float(value) for value in np.asarray(values_deg, dtype=float)),
            coords_A=coords_A,
            soft_energy_kj_mol=float("inf"),
            full_energy_kj_mol=float("inf"),
            score=float(clash_penalty) * (1.0 + delta * delta),
            hbond_metrics=hb,
            hbond_like_fraction=float(hb.like_fraction),
            hbond_geometric_fraction=float(hb.geometric_fraction),
            hbond_like_donor_occupancy_fraction=float(hb.like_donor_occupancy_fraction),
            hbond_geometric_donor_occupancy_fraction=float(
                hb.geometric_donor_occupancy_fraction
            ),
            hbond_like_satisfied_donors=int(hb.like_satisfied_donors),
            hbond_geometric_satisfied_donors=int(hb.geometric_satisfied_donors),
            hbond_family_metrics=family_metrics,
            hbond_family_min_like_fraction=float(family_min_like),
            hbond_family_min_geometric_fraction=float(family_min_geom),
            min_heavy_distance_A=float(dmin),
            class_min_distance_A=clipped_class_min,
            selector_aromatic_stacking_A=dict(stacking),
        )
        cache[key] = result
        return result

    soft_energy = float(engine.soft_energy_kj_mol(coords_A))
    use_full_energy = bool(
        engine.full_context is not None and objective_stage.use_full_energy_in_search
    )
    full_energy = (
        float(engine.full_energy_kj_mol(coords_A))
        if use_full_energy
        else float(soft_energy)
    )
    score = (
        float(objective_stage.weight_geom_occ)
        * (1.0 - float(hb.geometric_donor_occupancy_fraction))
        + float(objective_stage.weight_like_occ)
        * (1.0 - float(hb.like_donor_occupancy_fraction))
        + float(objective_stage.weight_geom_frac)
        * (1.0 - float(hb.geometric_fraction))
        + float(objective_stage.weight_family_min_geom)
        * (1.0 - float(family_min_geom))
        + float(objective_stage.weight_family_min_like)
        * (1.0 - float(family_min_like))
        + float(objective_stage.weight_soft_energy)
        * _relative_energy_term(
            soft_energy,
            baseline_soft_energy,
            float(objective_stage.energy_clip),
        )
        + float(objective_stage.weight_full_energy)
        * _relative_energy_term(
            full_energy,
            baseline_full_energy,
            float(objective_stage.energy_clip),
        )
    )
    result = _CandidateEval(
        values_deg=tuple(float(value) for value in np.asarray(values_deg, dtype=float)),
        coords_A=coords_A,
        soft_energy_kj_mol=float(soft_energy),
        full_energy_kj_mol=float(full_energy),
        score=float(score),
        hbond_metrics=hb,
        hbond_like_fraction=float(hb.like_fraction),
        hbond_geometric_fraction=float(hb.geometric_fraction),
        hbond_like_donor_occupancy_fraction=float(hb.like_donor_occupancy_fraction),
        hbond_geometric_donor_occupancy_fraction=float(
            hb.geometric_donor_occupancy_fraction
        ),
        hbond_like_satisfied_donors=int(hb.like_satisfied_donors),
        hbond_geometric_satisfied_donors=int(hb.geometric_satisfied_donors),
        hbond_family_metrics=family_metrics,
        hbond_family_min_like_fraction=float(family_min_like),
        hbond_family_min_geometric_fraction=float(family_min_geom),
        min_heavy_distance_A=float(dmin),
        class_min_distance_A=clipped_class_min,
        selector_aromatic_stacking_A=dict(stacking),
    )
    cache[key] = result
    return result


def _rank_network_candidates(candidates: Iterable[_CandidateEval]) -> _CandidateEval:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.hbond_family_min_geometric_fraction),
            -float(item.hbond_family_min_like_fraction),
            -float(item.hbond_geometric_donor_occupancy_fraction),
            -float(item.hbond_geometric_fraction),
            (
                float(item.hbond_metrics.mean_geometric_distance_A)
                if int(item.hbond_metrics.geometric_satisfied_pairs) > 0
                else float("inf")
            ),
            -float(item.hbond_like_donor_occupancy_fraction),
            -float(item.hbond_like_fraction),
            (
                float(item.hbond_metrics.mean_like_distance_A)
                if int(item.hbond_metrics.like_satisfied_pairs) > 0
                else float("inf")
            ),
            -float(item.min_heavy_distance_A),
            float(item.full_energy_kj_mol),
            float(item.soft_energy_kj_mol),
            float(item.score),
        ),
    )
    if not ranked:
        raise ValueError("No symmetry-network candidates available for ranking.")
    return ranked[0]


def _run_network_search(
    engine: _SymmetryOrderingEngine | _SymmetryBackboneRefinementEngine,
    *,
    selector: SelectorTemplate,
    objective_stage: _NetworkObjectiveStage,
    initial_values_deg: np.ndarray,
    baseline_soft_energy: float,
    baseline_full_energy: float,
    seed: int | None,
    maxiter: int,
    popsize: int,
    polish: bool,
    init_population: np.ndarray | None,
    rerank_population: bool,
) -> tuple[_CandidateEval, _CandidateEval, int, object | None]:
    eval_cache: dict[tuple[float, ...], _CandidateEval] = {}
    baseline_candidate = _evaluate_network_candidate(
        engine,
        selector=selector,
        objective_stage=objective_stage,
        values_deg=initial_values_deg,
        baseline_soft_energy=baseline_soft_energy,
        baseline_full_energy=baseline_full_energy,
        cache=eval_cache,
    )
    best_candidate = baseline_candidate
    evaluation_count = 1
    result = None

    if initial_values_deg.size > 0:
        bounds = [(-180.0, 180.0)] * int(initial_values_deg.size)

        def _objective(values_deg: np.ndarray) -> float:
            nonlocal evaluation_count
            evaluation_count += 1
            try:
                candidate = _evaluate_network_candidate(
                    engine,
                    selector=selector,
                    objective_stage=objective_stage,
                    values_deg=np.asarray(values_deg, dtype=float),
                    baseline_soft_energy=baseline_soft_energy,
                    baseline_full_energy=baseline_full_energy,
                    cache=eval_cache,
                )
                if not np.isfinite(candidate.score):
                    return float(1.0e12)
                return float(candidate.score)
            except Exception:
                return float(1.0e12)

        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=seed,
            maxiter=max(1, int(maxiter)),
            popsize=max(1, int(popsize)),
            polish=bool(polish),
            updating="deferred",
            workers=1,
            init=("latinhypercube" if init_population is None else init_population),
        )

        if bool(rerank_population):
            candidate_pool: list[_CandidateEval] = [baseline_candidate]
            population = getattr(result, "population", None)
            if population is not None:
                for values in np.asarray(population, dtype=float):
                    candidate_pool.append(
                        _evaluate_network_candidate(
                            engine,
                            selector=selector,
                            objective_stage=objective_stage,
                            values_deg=np.asarray(values, dtype=float),
                            baseline_soft_energy=baseline_soft_energy,
                            baseline_full_energy=baseline_full_energy,
                            cache=eval_cache,
                        )
                    )
            candidate_pool.append(
                _evaluate_network_candidate(
                    engine,
                    selector=selector,
                    objective_stage=objective_stage,
                    values_deg=np.asarray(result.x, dtype=float),
                    baseline_soft_energy=baseline_soft_energy,
                    baseline_full_energy=baseline_full_energy,
                    cache=eval_cache,
                )
            )
            best_candidate = _rank_network_candidates(candidate_pool)
        else:
            best_candidate = _evaluate_network_candidate(
                engine,
                selector=selector,
                objective_stage=objective_stage,
                values_deg=np.asarray(result.x, dtype=float),
                baseline_soft_energy=baseline_soft_energy,
                baseline_full_energy=baseline_full_energy,
                cache=eval_cache,
            )

    return baseline_candidate, best_candidate, evaluation_count, result


def optimize_symmetry_coupled_ordering(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    sites: Iterable[Site],
    dp: int,
    spec: OrderingSpec,
    grid: RotamerGridSpec | None = None,
    seed: int | None = None,
    *,
    runtime_params: RuntimeParams | None = None,
    work_dir: str | Path | None = None,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = None,
    mixing_rules_cfg: Mapping[str, object] | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    _require_forcefield_molecule(mol)

    if runtime_params is None:
        runtime_params = load_runtime_params(
            mol,
            selector_template=selector,
            work_dir=None if work_dir is None else Path(work_dir),
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

    if not spec.enabled:
        hb, dmin, class_min, stacking = _ordering_diagnostics(mol, selector, spec)
        symmetry_rmsd = selector_screw_symmetry_rmsd_from_mol(mol)
        applied_hbond_connectivity_policy = resolve_hbond_connectivity_policy(
            mol,
            selector,
            requested_policy=spec.hbond_connectivity_policy,
            requested_sites=sites,
        )
        return Chem.Mol(mol), {
            "enabled": False,
            "strategy": "symmetry_coupled",
            "objective": _ordering_objective_label(spec),
            "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
            "hbond_connectivity_policy_applied": str(applied_hbond_connectivity_policy),
            "baseline_energy_kj_mol": None,
            "stage1_nonbonded_mode": None,
            "stage2_nonbonded_mode": None,
            "full_stage_skipped": bool(spec.skip_full_stage),
            "final_stage_nonbonded_mode": None,
            "baseline_hbond_like_fraction": hb.like_fraction,
            "baseline_hbond_geometric_fraction": hb.geometric_fraction,
            "baseline_hbond_donor_count": hb.donor_count,
            "baseline_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "baseline_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "baseline_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "baseline_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "baseline_min_heavy_distance_A": dmin,
            "baseline_class_min_distance_A": {
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "baseline_selector_aromatic_stacking_A": dict(stacking),
            "baseline_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "initialization_mode": "disabled",
            "site_sweep_count": 0,
            "initial_energy_kj_mol": None,
            "initial_hbond_like_fraction": hb.like_fraction,
            "initial_hbond_geometric_fraction": hb.geometric_fraction,
            "initial_hbond_donor_count": hb.donor_count,
            "initial_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "initial_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "initial_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "initial_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "initial_selector_aromatic_stacking_A": dict(stacking),
            "initial_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "final_energy_kj_mol": None,
            "final_score": None,
            "final_hbond_like_fraction": hb.like_fraction,
            "final_hbond_geometric_fraction": hb.geometric_fraction,
            "final_hbond_donor_count": hb.donor_count,
            "final_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "final_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "final_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "final_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "final_min_heavy_distance_A": dmin,
            "final_class_min_distance_A": {
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "final_selector_aromatic_stacking_A": dict(stacking),
            "final_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "initial_pose_by_site": {},
            "selected_pose_by_site": {},
        }

    engine = _build_engine(
        mol,
        selector=selector,
        sites=[str(site) for site in sites],
        dp=dp,
        spec=spec,
        grid=grid,
        include_anchor_dihedrals=False,
        runtime_params=runtime_params,
        mixing_rules_cfg=mixing_rules_cfg,
    )
    initial_pose = engine.initial_dihedrals_deg()
    initial_values_deg = np.asarray(
        [
            initial_pose[term.site][term.name]
            for term in engine.active_dihedrals
        ],
        dtype=float,
    )
    baseline_coords_A = engine.build_coords(initial_values_deg)
    baseline_mol = update_rdkit_coords(
        engine.mol,
        (baseline_coords_A / 10.0) * unit.nanometer,
    )
    baseline_soft_energy = engine.soft_energy_kj_mol(baseline_coords_A)
    baseline_full_energy = (
        baseline_soft_energy
        if bool(spec.skip_full_stage)
        else engine.full_energy_kj_mol(baseline_coords_A)
    )
    best_values_deg = initial_values_deg.copy()
    best_coords_A = baseline_coords_A
    evaluation_count = 1

    if initial_values_deg.size > 0:
        bounds = [(-180.0, 180.0)] * int(initial_values_deg.size)

        def _objective(values_deg: np.ndarray) -> float:
            nonlocal evaluation_count
            evaluation_count += 1
            try:
                coords_A = engine.build_coords(np.asarray(values_deg, dtype=float))
                energy = float(engine.soft_energy_kj_mol(coords_A))
                if not np.isfinite(energy):
                    return float(1.0e12)
                return energy
            except Exception:
                return float(1.0e12)

        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=seed,
            maxiter=max(1, int(spec.symmetry_maxiter)),
            popsize=max(1, int(spec.symmetry_popsize)),
            polish=bool(spec.symmetry_polish),
            updating="deferred",
            workers=1,
        )
        best_values_deg = np.asarray(result.x, dtype=float)
        best_coords_A = engine.build_coords(best_values_deg)

    final_mol = update_rdkit_coords(
        engine.mol,
        (best_coords_A / 10.0) * unit.nanometer,
    )
    final_soft_energy = engine.soft_energy_kj_mol(best_coords_A)
    final_full_energy = (
        final_soft_energy
        if bool(spec.skip_full_stage)
        else engine.full_energy_kj_mol(best_coords_A)
    )

    baseline_hb, baseline_dmin, baseline_class_min, baseline_stacking = _ordering_diagnostics(
        baseline_mol,
        selector,
        spec,
    )
    final_hb, final_dmin, final_class_min, final_stacking = _ordering_diagnostics(
        final_mol,
        selector,
        spec,
    )
    baseline_symmetry = selector_screw_symmetry_rmsd_from_mol(baseline_mol)
    final_symmetry = selector_screw_symmetry_rmsd_from_mol(final_mol)

    selected_pose = engine.measure_dihedrals_deg(best_coords_A)
    summary: Dict[str, object] = {
        "enabled": True,
        "strategy": "symmetry_coupled",
        "objective": _ordering_objective_label(spec),
        "search_objective": "soft_single_point_energy_kj_mol",
        "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
        "hbond_connectivity_policy_applied": str(
            resolve_hbond_connectivity_policy(
                final_mol,
                selector,
                requested_policy=spec.hbond_connectivity_policy,
                requested_sites=engine.sites,
            )
        ),
        "stage1_nonbonded_mode": engine.soft_nonbonded_mode,
        "soft_exception_summary": dict(engine.soft_exception_summary),
        "stage2_nonbonded_mode": engine.full_nonbonded_mode,
        "full_stage_skipped": bool(spec.skip_full_stage),
        "final_stage_nonbonded_mode": (
            engine.soft_nonbonded_mode
            if bool(spec.skip_full_stage)
            else engine.full_nonbonded_mode
        ),
        "repeat_residues": 1,
        "helix_repeat_residues": (
            int(mol.GetIntProp("_poly_csp_helix_repeat_residues"))
            if mol.HasProp("_poly_csp_helix_repeat_residues")
            else None
        ),
        "candidate_count": (
            int(spec.symmetry_popsize) * max(1, len(engine.active_dihedrals))
        ),
        "evaluation_count": evaluation_count,
        "initialization_mode": "reference_residue_projection",
        "site_sweep_count": 0,
        "seed": seed,
        "active_symmetry_dof": len(engine.active_dihedrals),
        "symmetry_mode": "exact_screw_projection",
        "baseline_energy_kj_mol": float(baseline_full_energy),
        "baseline_stage1_energies_kj_mol": [float(baseline_soft_energy)],
        "baseline_stage2_energies_kj_mol": (
            [float(baseline_soft_energy)]
            if bool(spec.skip_full_stage)
            else [float(baseline_full_energy)]
        ),
        "baseline_hbond_like_fraction": baseline_hb.like_fraction,
        "baseline_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "baseline_hbond_donor_count": baseline_hb.donor_count,
        "baseline_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "baseline_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "baseline_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "baseline_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "baseline_min_heavy_distance_A": float(baseline_dmin),
        "baseline_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in baseline_class_min.items()
        },
        "baseline_selector_aromatic_stacking_A": dict(baseline_stacking),
        "baseline_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "initial_energy_kj_mol": float(baseline_full_energy),
        "initial_hbond_like_fraction": baseline_hb.like_fraction,
        "initial_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "initial_hbond_donor_count": baseline_hb.donor_count,
        "initial_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "initial_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "initial_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "initial_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "initial_selector_aromatic_stacking_A": dict(baseline_stacking),
        "initial_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "final_energy_kj_mol": float(final_full_energy),
        "final_score": -float(final_full_energy),
        "final_stage1_energies_kj_mol": [float(final_soft_energy)],
        "final_stage2_energies_kj_mol": (
            [float(final_soft_energy)]
            if bool(spec.skip_full_stage)
            else [float(final_full_energy)]
        ),
        "final_hbond_like_fraction": final_hb.like_fraction,
        "final_hbond_geometric_fraction": final_hb.geometric_fraction,
        "final_hbond_donor_count": final_hb.donor_count,
        "final_hbond_like_satisfied_donors": final_hb.like_satisfied_donors,
        "final_hbond_geometric_satisfied_donors": final_hb.geometric_satisfied_donors,
        "final_hbond_like_donor_occupancy_fraction": (
            final_hb.like_donor_occupancy_fraction
        ),
        "final_hbond_geometric_donor_occupancy_fraction": (
            final_hb.geometric_donor_occupancy_fraction
        ),
        "final_min_heavy_distance_A": float(final_dmin),
        "final_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in final_class_min.items()
        },
        "final_selector_aromatic_stacking_A": dict(final_stacking),
        "final_selector_symmetry_rmsd_A": float(final_symmetry),
        "soft_search_energy_kj_mol": float(final_soft_energy),
        "initial_pose_by_site": _symmetry_pose_summary(initial_pose),
        "selected_pose_by_site": _symmetry_pose_summary(selected_pose),
        "optimized_dihedrals_by_site": {
            site: {name: float(value) for name, value in sorted(values.items())}
            for site, values in selected_pose.items()
        },
    }
    return final_mol, summary


def optimize_symmetry_network_ordering(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    sites: Iterable[Site],
    dp: int,
    spec: OrderingSpec,
    grid: RotamerGridSpec | None = None,
    seed: int | None = None,
    *,
    runtime_params: RuntimeParams | None = None,
    work_dir: str | Path | None = None,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = None,
    mixing_rules_cfg: Mapping[str, object] | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    _require_forcefield_molecule(mol)

    if runtime_params is None:
        runtime_params = load_runtime_params(
            mol,
            selector_template=selector,
            work_dir=None if work_dir is None else Path(work_dir),
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

    if not spec.enabled:
        hb, dmin, class_min, stacking = _ordering_diagnostics(mol, selector, spec)
        symmetry_rmsd = selector_screw_symmetry_rmsd_from_mol(mol)
        applied_hbond_connectivity_policy = resolve_hbond_connectivity_policy(
            mol,
            selector,
            requested_policy=spec.hbond_connectivity_policy,
            requested_sites=sites,
        )
        return Chem.Mol(mol), {
            "enabled": False,
            "strategy": "symmetry_network",
            "objective": _ordering_objective_label(spec),
            "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
            "hbond_connectivity_policy_applied": str(applied_hbond_connectivity_policy),
            "baseline_energy_kj_mol": None,
            "stage1_nonbonded_mode": None,
            "stage2_nonbonded_mode": None,
            "full_stage_skipped": bool(spec.skip_full_stage),
            "final_stage_nonbonded_mode": None,
            "baseline_hbond_like_fraction": hb.like_fraction,
            "baseline_hbond_geometric_fraction": hb.geometric_fraction,
            "baseline_hbond_donor_count": hb.donor_count,
            "baseline_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "baseline_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "baseline_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "baseline_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "baseline_min_heavy_distance_A": dmin,
            "baseline_class_min_distance_A": {
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "baseline_selector_aromatic_stacking_A": dict(stacking),
            "baseline_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "initialization_mode": "disabled",
            "site_sweep_count": 0,
            "initial_energy_kj_mol": None,
            "initial_hbond_like_fraction": hb.like_fraction,
            "initial_hbond_geometric_fraction": hb.geometric_fraction,
            "initial_hbond_donor_count": hb.donor_count,
            "initial_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "initial_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "initial_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "initial_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "initial_selector_aromatic_stacking_A": dict(stacking),
            "initial_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "final_energy_kj_mol": None,
            "final_score": None,
            "final_hbond_like_fraction": hb.like_fraction,
            "final_hbond_geometric_fraction": hb.geometric_fraction,
            "final_hbond_donor_count": hb.donor_count,
            "final_hbond_like_satisfied_donors": hb.like_satisfied_donors,
            "final_hbond_geometric_satisfied_donors": hb.geometric_satisfied_donors,
            "final_hbond_like_donor_occupancy_fraction": hb.like_donor_occupancy_fraction,
            "final_hbond_geometric_donor_occupancy_fraction": hb.geometric_donor_occupancy_fraction,
            "final_min_heavy_distance_A": dmin,
            "final_class_min_distance_A": {
                key: float(value) if np.isfinite(value) else None
                for key, value in class_min.items()
            },
            "final_selector_aromatic_stacking_A": dict(stacking),
            "final_selector_symmetry_rmsd_A": float(symmetry_rmsd),
            "initial_pose_by_site": {},
            "selected_pose_by_site": {},
        }

    engine = _build_engine(
        mol,
        selector=selector,
        sites=[str(site) for site in sites],
        dp=dp,
        spec=spec,
        grid=grid,
        include_anchor_dihedrals=True,
        runtime_params=runtime_params,
        mixing_rules_cfg=mixing_rules_cfg,
    )
    initial_pose = engine.initial_dihedrals_deg()
    initial_values_deg = np.asarray(
        [initial_pose[term.site][term.name] for term in engine.active_dihedrals],
        dtype=float,
    )
    baseline_coords_A = engine.build_coords(initial_values_deg)
    baseline_soft_energy = float(engine.soft_energy_kj_mol(baseline_coords_A))
    baseline_full_energy = (
        baseline_soft_energy
        if engine.full_context is None
        else float(engine.full_energy_kj_mol(baseline_coords_A))
    )
    baseline_candidate = _evaluate_network_candidate(
        engine,
        selector=selector,
        objective_stage=_cleanup_objective_stage(spec),
        values_deg=initial_values_deg,
        baseline_soft_energy=baseline_soft_energy,
        baseline_full_energy=baseline_full_energy,
        cache={},
    )

    capture_enabled = bool(spec.symmetry_network_capture_enabled)
    cleanup_enabled = bool(spec.symmetry_network_cleanup_enabled)
    capture_stage = _capture_objective_stage(spec)
    cleanup_stage = _cleanup_objective_stage(spec)
    capture_engine = None
    capture_best_candidate = None
    capture_result = None
    capture_evaluation_count = 0
    capture_candidate_count = 0
    selector_search_initialization_mode = (
        "rotamer_grid_seeded_de"
        if bool(spec.symmetry_init_from_rotamer_grid)
        else "reference_residue_projection"
    )
    cleanup_result = None
    cleanup_evaluation_count = 0
    cleanup_candidate_count = 0
    cleanup_applied = False

    if capture_enabled:
        capture_spec = _capture_ordering_spec(spec)
        capture_engine = _build_engine(
            mol,
            selector=selector,
            sites=[str(site) for site in sites],
            dp=dp,
            spec=capture_spec,
            grid=grid,
            include_anchor_dihedrals=True,
            runtime_params=runtime_params,
            mixing_rules_cfg=mixing_rules_cfg,
        )
        capture_initial_pose = capture_engine.initial_dihedrals_deg()
        capture_initial_values = np.asarray(
            [capture_initial_pose[term.site][term.name] for term in capture_engine.active_dihedrals],
            dtype=float,
        )
        capture_base_coords_A = capture_engine.build_coords(capture_initial_values)
        capture_baseline_soft = float(capture_engine.soft_energy_kj_mol(capture_base_coords_A))
        capture_init_population = (
            _initial_population_from_rotamers(
                capture_engine,
                initial_values_deg=capture_initial_values,
                popsize=int(spec.symmetry_popsize),
                jitter_deg=float(spec.symmetry_init_jitter_deg),
                seed=seed,
            )
            if bool(spec.symmetry_init_from_rotamer_grid)
            else None
        )
        capture_candidate_count = int(spec.symmetry_popsize) * max(
            1,
            len(capture_engine.active_dihedrals),
        )
        _, capture_best_candidate, capture_evaluation_count, capture_result = _run_network_search(
            capture_engine,
            selector=selector,
            objective_stage=capture_stage,
            initial_values_deg=capture_initial_values,
            baseline_soft_energy=capture_baseline_soft,
            baseline_full_energy=capture_baseline_soft,
            seed=seed,
            maxiter=int(spec.symmetry_maxiter),
            popsize=int(spec.symmetry_popsize),
            polish=bool(spec.symmetry_polish),
            init_population=capture_init_population,
            rerank_population=bool(spec.symmetry_network_rerank_population),
        )
        selector_search_initialization_mode = (
            "staged_capture_rotamer_grid_seeded_de"
            if bool(spec.symmetry_init_from_rotamer_grid)
            else "staged_capture_reference_projection"
        )

    if capture_enabled and capture_best_candidate is not None:
        cleanup_engine = _rebase_engine(
            engine,
            base_coords_A=np.asarray(capture_best_candidate.coords_A, dtype=float),
        )
        cleanup_initial_pose = cleanup_engine.initial_dihedrals_deg()
        cleanup_initial_values = np.asarray(
            [cleanup_initial_pose[term.site][term.name] for term in cleanup_engine.active_dihedrals],
            dtype=float,
        )
        cleanup_baseline_soft = float(
            cleanup_engine.soft_energy_kj_mol(cleanup_engine.base_coords_A)
        )
        cleanup_baseline_full = (
            cleanup_baseline_soft
            if cleanup_engine.full_context is None
            else float(cleanup_engine.full_energy_kj_mol(cleanup_engine.base_coords_A))
        )
        if cleanup_enabled:
            cleanup_applied = True
            cleanup_candidate_count = int(spec.symmetry_network_cleanup_popsize) * max(
                1,
                len(cleanup_engine.active_dihedrals),
            )
            cleanup_init_population = _initial_population_from_rotamers(
                cleanup_engine,
                initial_values_deg=cleanup_initial_values,
                popsize=int(spec.symmetry_network_cleanup_popsize),
                jitter_deg=float(spec.symmetry_network_cleanup_init_jitter_deg),
                seed=seed,
            )
            _, best_candidate, cleanup_evaluation_count, cleanup_result = _run_network_search(
                cleanup_engine,
                selector=selector,
                objective_stage=cleanup_stage,
                initial_values_deg=cleanup_initial_values,
                baseline_soft_energy=cleanup_baseline_soft,
                baseline_full_energy=cleanup_baseline_full,
                seed=seed,
                maxiter=int(spec.symmetry_network_cleanup_maxiter),
                popsize=int(spec.symmetry_network_cleanup_popsize),
                polish=bool(spec.symmetry_network_cleanup_polish),
                init_population=cleanup_init_population,
                rerank_population=bool(spec.symmetry_network_rerank_population),
            )
        else:
            best_candidate = _evaluate_network_candidate(
                cleanup_engine,
                selector=selector,
                objective_stage=cleanup_stage,
                values_deg=cleanup_initial_values,
                baseline_soft_energy=cleanup_baseline_soft,
                baseline_full_energy=cleanup_baseline_full,
                cache={},
            )
    else:
        init_population = (
            _initial_population_from_rotamers(
                engine,
                initial_values_deg=initial_values_deg,
                popsize=int(spec.symmetry_popsize),
                jitter_deg=float(spec.symmetry_init_jitter_deg),
                seed=seed,
            )
            if bool(spec.symmetry_init_from_rotamer_grid)
            else None
        )
        cleanup_candidate_count = int(spec.symmetry_popsize) * max(1, len(engine.active_dihedrals))
        _, best_candidate, cleanup_evaluation_count, cleanup_result = _run_network_search(
            engine,
            selector=selector,
            objective_stage=cleanup_stage,
            initial_values_deg=initial_values_deg,
            baseline_soft_energy=baseline_soft_energy,
            baseline_full_energy=baseline_full_energy,
            seed=seed,
            maxiter=int(spec.symmetry_maxiter),
            popsize=int(spec.symmetry_popsize),
            polish=bool(spec.symmetry_polish),
            init_population=init_population,
            rerank_population=bool(spec.symmetry_network_rerank_population),
        )

    evaluation_count = int(capture_evaluation_count) + int(cleanup_evaluation_count)
    result = cleanup_result

    primary_best_candidate = best_candidate
    refinement_engine, backbone_refinement_skipped_reason = _build_backbone_refinement_engine(
        engine,
        selector=selector,
        grid=grid,
        dp=dp,
        base_coords_A=np.asarray(best_candidate.coords_A, dtype=float),
    )
    backbone_refinement_applied = refinement_engine is not None
    backbone_refinement_selected = False
    backbone_refinement_result = None
    backbone_refinement_eval_count = 0
    backbone_refinement_candidate_count = 0
    backbone_initial_dihedrals: dict[str, float] = {}
    backbone_final_dihedrals: dict[str, float] = {}
    active_backbone_dihedral_names: list[str] = []
    active_backbone_total_dof = 0

    if refinement_engine is not None:
        active_backbone_dihedral_names = [
            term.name
            for term in refinement_engine.active_terms
            if term.kind == "backbone"
        ]
        active_backbone_total_dof = len(refinement_engine.active_terms)
        backbone_refinement_candidate_count = (
            int(spec.symmetry_backbone_popsize) * max(1, active_backbone_total_dof)
        )
        backbone_initial_dihedrals = refinement_engine.backbone_dihedrals_deg(
            refinement_engine.base_coords_A
        )
        refinement_initial_values = refinement_engine.initial_values_deg()
        refinement_baseline_soft = float(
            refinement_engine.soft_energy_kj_mol(refinement_engine.base_coords_A)
        )
        refinement_baseline_full = (
            refinement_baseline_soft
            if refinement_engine.full_context is None
            else float(refinement_engine.full_energy_kj_mol(refinement_engine.base_coords_A))
        )
        init_population = _initial_population_from_coupled_terms(
            refinement_engine.active_terms,
            initial_values_deg=refinement_initial_values,
            popsize=int(spec.symmetry_backbone_popsize),
            jitter_deg=float(spec.symmetry_backbone_init_jitter_deg),
            seed=seed,
        )
        _, refinement_best_candidate, backbone_refinement_eval_count, backbone_refinement_result = _run_network_search(
            refinement_engine,
            selector=selector,
            objective_stage=cleanup_stage,
            initial_values_deg=refinement_initial_values,
            baseline_soft_energy=refinement_baseline_soft,
            baseline_full_energy=refinement_baseline_full,
            seed=seed,
            maxiter=int(spec.symmetry_backbone_maxiter),
            popsize=int(spec.symmetry_backbone_popsize),
            polish=bool(spec.symmetry_backbone_polish),
            init_population=init_population,
            rerank_population=bool(spec.symmetry_network_rerank_population),
        )

        best_candidate = _rank_network_candidates([primary_best_candidate, refinement_best_candidate])
        backbone_refinement_selected = best_candidate is not primary_best_candidate
        backbone_final_dihedrals = refinement_engine.backbone_dihedrals_deg(
            np.asarray(best_candidate.coords_A, dtype=float)
        )

    best_coords_A = np.asarray(best_candidate.coords_A, dtype=float)
    final_mol = update_rdkit_coords(engine.mol, (best_coords_A / 10.0) * unit.nanometer)
    final_soft_energy = float(engine.soft_energy_kj_mol(best_coords_A))
    final_full_energy = (
        final_soft_energy
        if engine.full_context is None
        else float(engine.full_energy_kj_mol(best_coords_A))
    )

    baseline_mol = update_rdkit_coords(engine.mol, (baseline_coords_A / 10.0) * unit.nanometer)
    baseline_hb, baseline_dmin, baseline_class_min, baseline_stacking = _ordering_diagnostics(
        baseline_mol,
        selector,
        spec,
    )
    final_hb, final_dmin, final_class_min, final_stacking = _ordering_diagnostics(
        final_mol,
        selector,
        spec,
    )
    baseline_symmetry = selector_screw_symmetry_rmsd_from_mol(baseline_mol)
    final_symmetry = selector_screw_symmetry_rmsd_from_mol(final_mol)

    if refinement_engine is not None and bool(spec.symmetry_backbone_reoptimize_selectors):
        selected_pose = refinement_engine.selector_dihedrals_deg(best_coords_A)
    else:
        selected_pose = engine.measure_dihedrals_deg(best_coords_A)
    active_anchor_dihedrals = sorted(
        {
            term.name
            for term in engine.active_dihedrals
            if term.name in selector.anchor_dihedrals
        }
    )
    applied_hbond_connectivity_policy = resolve_hbond_connectivity_policy(
        engine.mol,
        selector,
        requested_policy=spec.hbond_connectivity_policy,
        requested_sites=engine.sites,
    )
    selector_stage_selection_method = (
        (
            "population_rerank"
            if bool(spec.symmetry_network_rerank_population)
            else "de_best"
        )
        if not bool(capture_enabled)
        else (
            "cleanup_population_rerank"
            if bool(cleanup_applied) and bool(spec.symmetry_network_rerank_population)
            else (
                "cleanup_de_best"
                if bool(cleanup_applied)
                else (
                    "capture_population_rerank"
                    if bool(spec.symmetry_network_rerank_population)
                    else "capture_de_best"
                )
            )
        )
    )
    summary: Dict[str, object] = {
        "enabled": True,
        "strategy": "symmetry_network",
        "objective": _ordering_objective_label(spec),
        "search_objective": (
            "staged_network_first_symmetry_score"
            if bool(capture_enabled)
            else "network_first_symmetry_score"
        ),
        "stage1_nonbonded_mode": engine.soft_nonbonded_mode,
        "soft_exception_summary": dict(engine.soft_exception_summary),
        "stage2_nonbonded_mode": engine.full_nonbonded_mode,
        "full_stage_skipped": bool(spec.skip_full_stage),
        "final_stage_nonbonded_mode": (
            engine.soft_nonbonded_mode
            if bool(spec.skip_full_stage)
            else engine.full_nonbonded_mode
        ),
        "repeat_residues": 1,
        "helix_repeat_residues": (
            int(mol.GetIntProp("_poly_csp_helix_repeat_residues"))
            if mol.HasProp("_poly_csp_helix_repeat_residues")
            else None
        ),
        "candidate_count": int(capture_candidate_count + cleanup_candidate_count),
        "evaluation_count": evaluation_count,
        "initialization_mode": selector_search_initialization_mode,
        "site_sweep_count": 0,
        "seed": seed,
        "active_symmetry_dof": len(engine.active_dihedrals),
        "active_anchor_dihedral_names": active_anchor_dihedrals,
        "active_backbone_dihedral_names": active_backbone_dihedral_names,
        "symmetry_mode": "exact_screw_projection",
        "hbond_connectivity_policy_requested": str(spec.hbond_connectivity_policy),
        "hbond_connectivity_policy_applied": str(applied_hbond_connectivity_policy),
        "network_capture_enabled": bool(capture_enabled),
        "network_capture_applied": bool(capture_enabled),
        "network_capture_candidate_count": int(capture_candidate_count),
        "network_capture_evaluation_count": int(capture_evaluation_count),
        "network_capture_stage_nonbonded_mode": (
            None if capture_engine is None else capture_engine.soft_nonbonded_mode
        ),
        "network_capture_score_weights": {
            "geom_occ": float(spec.symmetry_network_capture_weight_geom_occ),
            "like_occ": float(spec.symmetry_network_capture_weight_like_occ),
            "geom_frac": float(spec.symmetry_network_capture_weight_geom_frac),
            "family_min_geom": float(spec.symmetry_network_capture_weight_family_min_geom),
            "family_min_like": float(spec.symmetry_network_capture_weight_family_min_like),
            "soft_energy": float(spec.symmetry_network_capture_weight_soft_energy),
            "full_energy": 0.0,
        },
        "network_capture_clash_settings": {
            "soft_repulsion_scale": float(spec.symmetry_network_capture_soft_repulsion_scale),
            "min_heavy_distance_A": float(spec.symmetry_network_capture_min_heavy_distance_A),
            "clash_penalty": float(spec.symmetry_network_capture_clash_penalty),
            "energy_clip": float(spec.symmetry_network_capture_energy_clip),
            "use_full_energy_in_search": False,
            "rerank_population": bool(spec.symmetry_network_rerank_population),
        },
        "network_score_weights": {
            "geom_occ": float(spec.symmetry_network_weight_geom_occ),
            "like_occ": float(spec.symmetry_network_weight_like_occ),
            "geom_frac": float(spec.symmetry_network_weight_geom_frac),
            "family_min_geom": float(spec.symmetry_network_weight_family_min_geom),
            "family_min_like": float(spec.symmetry_network_weight_family_min_like),
            "soft_energy": float(spec.symmetry_network_weight_soft_energy),
            "full_energy": float(spec.symmetry_network_weight_full_energy),
        },
        "network_score_clash_settings": {
            "min_heavy_distance_A": float(spec.symmetry_network_min_heavy_distance_A),
            "clash_penalty": float(spec.symmetry_network_clash_penalty),
            "energy_clip": float(spec.symmetry_network_energy_clip),
            "use_full_energy_in_search": bool(spec.symmetry_network_use_full_energy_in_search),
            "rerank_population": bool(spec.symmetry_network_rerank_population),
        },
        "network_cleanup_enabled": bool(cleanup_enabled),
        "network_cleanup_applied": bool(cleanup_applied or not capture_enabled),
        "network_cleanup_candidate_count": int(cleanup_candidate_count),
        "network_cleanup_evaluation_count": int(cleanup_evaluation_count),
        "network_cleanup_budget": {
            "maxiter": (
                int(spec.symmetry_network_cleanup_maxiter)
                if bool(capture_enabled)
                else int(spec.symmetry_maxiter)
            ),
            "popsize": (
                int(spec.symmetry_network_cleanup_popsize)
                if bool(capture_enabled)
                else int(spec.symmetry_popsize)
            ),
            "polish": (
                bool(spec.symmetry_network_cleanup_polish)
                if bool(capture_enabled)
                else bool(spec.symmetry_polish)
            ),
            "init_jitter_deg": (
                float(spec.symmetry_network_cleanup_init_jitter_deg)
                if bool(capture_enabled)
                else float(spec.symmetry_init_jitter_deg)
            ),
        },
        "final_selection_method": (
            "backbone_refinement_population_rerank"
            if backbone_refinement_selected and bool(spec.symmetry_network_rerank_population)
            else (
                "backbone_refinement_de_best"
                if backbone_refinement_selected
                else selector_stage_selection_method
            )
        ),
        "backbone_refinement_enabled": bool(spec.symmetry_backbone_refine_enabled),
        "backbone_refinement_applied": bool(backbone_refinement_applied),
        "backbone_refinement_selected": bool(backbone_refinement_selected),
        "backbone_refinement_skipped_reason": backbone_refinement_skipped_reason,
        "backbone_refinement_reoptimize_selectors": bool(
            spec.symmetry_backbone_reoptimize_selectors
        ),
        "backbone_refinement_active_dof": int(active_backbone_total_dof),
        "backbone_refinement_candidate_count": int(backbone_refinement_candidate_count),
        "backbone_refinement_evaluation_count": int(backbone_refinement_eval_count),
        "backbone_refinement_initial_backbone_dihedrals_deg": {
            name: float(value) for name, value in sorted(backbone_initial_dihedrals.items())
        },
        "backbone_refinement_final_backbone_dihedrals_deg": {
            name: float(value) for name, value in sorted(backbone_final_dihedrals.items())
        },
        "baseline_energy_kj_mol": float(baseline_full_energy),
        "baseline_stage1_energies_kj_mol": [float(baseline_soft_energy)],
        "baseline_stage2_energies_kj_mol": (
            [float(baseline_soft_energy)]
            if engine.full_context is None
            else [float(baseline_full_energy)]
        ),
        "baseline_hbond_like_fraction": baseline_hb.like_fraction,
        "baseline_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "baseline_hbond_donor_count": baseline_hb.donor_count,
        "baseline_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "baseline_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "baseline_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "baseline_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "baseline_hbond_family_metrics": _hbond_family_metrics_summary(
            baseline_candidate.hbond_family_metrics
        ),
        "baseline_hbond_family_min_like_fraction": float(
            baseline_candidate.hbond_family_min_like_fraction
        ),
        "baseline_hbond_family_min_geometric_fraction": float(
            baseline_candidate.hbond_family_min_geometric_fraction
        ),
        "baseline_min_heavy_distance_A": float(baseline_dmin),
        "baseline_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in baseline_class_min.items()
        },
        "baseline_selector_aromatic_stacking_A": dict(baseline_stacking),
        "baseline_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "initial_energy_kj_mol": float(baseline_full_energy),
        "initial_hbond_like_fraction": baseline_hb.like_fraction,
        "initial_hbond_geometric_fraction": baseline_hb.geometric_fraction,
        "initial_hbond_donor_count": baseline_hb.donor_count,
        "initial_hbond_like_satisfied_donors": baseline_hb.like_satisfied_donors,
        "initial_hbond_geometric_satisfied_donors": baseline_hb.geometric_satisfied_donors,
        "initial_hbond_like_donor_occupancy_fraction": (
            baseline_hb.like_donor_occupancy_fraction
        ),
        "initial_hbond_geometric_donor_occupancy_fraction": (
            baseline_hb.geometric_donor_occupancy_fraction
        ),
        "initial_hbond_family_metrics": _hbond_family_metrics_summary(
            baseline_candidate.hbond_family_metrics
        ),
        "initial_hbond_family_min_like_fraction": float(
            baseline_candidate.hbond_family_min_like_fraction
        ),
        "initial_hbond_family_min_geometric_fraction": float(
            baseline_candidate.hbond_family_min_geometric_fraction
        ),
        "initial_selector_aromatic_stacking_A": dict(baseline_stacking),
        "initial_selector_symmetry_rmsd_A": float(baseline_symmetry),
        "final_energy_kj_mol": float(final_full_energy),
        "final_score": -float(best_candidate.score),
        "final_stage1_energies_kj_mol": [float(final_soft_energy)],
        "final_stage2_energies_kj_mol": (
            [float(final_soft_energy)]
            if engine.full_context is None
            else [float(final_full_energy)]
        ),
        "final_hbond_like_fraction": final_hb.like_fraction,
        "final_hbond_geometric_fraction": final_hb.geometric_fraction,
        "final_hbond_donor_count": final_hb.donor_count,
        "final_hbond_like_satisfied_donors": final_hb.like_satisfied_donors,
        "final_hbond_geometric_satisfied_donors": final_hb.geometric_satisfied_donors,
        "final_hbond_like_donor_occupancy_fraction": (
            final_hb.like_donor_occupancy_fraction
        ),
        "final_hbond_geometric_donor_occupancy_fraction": (
            final_hb.geometric_donor_occupancy_fraction
        ),
        "final_hbond_family_metrics": _hbond_family_metrics_summary(
            best_candidate.hbond_family_metrics
        ),
        "final_hbond_family_min_like_fraction": float(
            best_candidate.hbond_family_min_like_fraction
        ),
        "final_hbond_family_min_geometric_fraction": float(
            best_candidate.hbond_family_min_geometric_fraction
        ),
        "final_min_heavy_distance_A": float(final_dmin),
        "final_class_min_distance_A": {
            key: float(value) if np.isfinite(value) else None
            for key, value in final_class_min.items()
        },
        "final_selector_aromatic_stacking_A": dict(final_stacking),
        "final_selector_symmetry_rmsd_A": float(final_symmetry),
        "soft_search_energy_kj_mol": float(best_candidate.soft_energy_kj_mol),
        "initial_pose_by_site": _symmetry_pose_summary(initial_pose),
        "selected_pose_by_site": _symmetry_pose_summary(selected_pose),
        "optimized_dihedrals_by_site": {
            site: {name: float(value) for name, value in sorted(values.items())}
            for site, values in selected_pose.items()
        },
    }
    if capture_best_candidate is not None:
        summary["network_capture_hbond_family_metrics"] = _hbond_family_metrics_summary(
            capture_best_candidate.hbond_family_metrics
        )
        summary["network_capture_hbond_family_min_geometric_fraction"] = float(
            capture_best_candidate.hbond_family_min_geometric_fraction
        )
        summary["network_capture_hbond_family_min_like_fraction"] = float(
            capture_best_candidate.hbond_family_min_like_fraction
        )
        summary["network_capture_min_heavy_distance_A"] = float(
            capture_best_candidate.min_heavy_distance_A
        )
        summary["network_capture_score"] = float(capture_best_candidate.score)
    if capture_result is not None:
        summary["network_capture_result_fun"] = float(capture_result.fun)
    if result is not None:
        summary["de_result_fun"] = float(result.fun)
        summary["network_cleanup_result_fun"] = float(result.fun)
    if backbone_refinement_result is not None:
        summary["backbone_refinement_result_fun"] = float(backbone_refinement_result.fun)
    return final_mol, summary
