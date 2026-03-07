"""Forcefield-domain mapping from PolyCSP backbone atoms to GLYCAM identities."""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from poly_csp.forcefield.glycam import (
    GlycamParams,
    ResidueRole,
    glycam_residue_roles_for_dp,
)


_GLYCAM_HYDROGEN_ALIASES = {
    "HO1": "H1O",
    "HO2": "H2O",
    "HO3": "H3O",
    "HO4": "H4O",
    "HO6": "H6O",
}

_SUBSTITUTED_SITE_ALLOWED_MISSING = {
    "O2": "H2O",
    "O3": "H3O",
    "O6": "H6O",
}


@dataclass(frozen=True)
class GlycamAtomAssignment:
    atom_index: int
    residue_index: int
    residue_role: ResidueRole
    glycam_residue_name: str
    generic_atom_name: str
    glycam_atom_name: str


@dataclass(frozen=True)
class GlycamMappingResult:
    polymer: str
    representation: str
    end_mode: str
    assignments: tuple[GlycamAtomAssignment, ...]


def _require_mol_prop(mol: Chem.Mol, name: str) -> str:
    if not mol.HasProp(name):
        raise ValueError(f"Forcefield-domain molecule is missing required property {name}.")
    return str(mol.GetProp(name))


def _parent_backbone_atom(mol: Chem.Mol, atom: Chem.Atom) -> Chem.Atom:
    if atom.GetAtomicNum() != 1:
        return atom
    if not atom.HasProp("_poly_csp_parent_heavy_idx"):
        raise ValueError(
            f"Hydrogen atom {atom.GetIdx()} is missing _poly_csp_parent_heavy_idx."
        )
    return mol.GetAtomWithIdx(int(atom.GetIntProp("_poly_csp_parent_heavy_idx")))


def _generic_to_glycam_name(atom_name: str) -> str:
    return _GLYCAM_HYDROGEN_ALIASES.get(atom_name, atom_name)


def _allowed_missing_hydrogens_by_residue(mol: Chem.Mol) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_manifest_source"):
            continue
        if atom.GetProp("_poly_csp_manifest_source") != "backbone":
            continue
        if not atom.HasProp("_poly_csp_atom_name") or not atom.HasProp("_poly_csp_residue_index"):
            continue
        atom_name = atom.GetProp("_poly_csp_atom_name")
        if atom_name not in _SUBSTITUTED_SITE_ALLOWED_MISSING:
            continue
        residue_index = int(atom.GetIntProp("_poly_csp_residue_index"))
        if any(
            nbr.GetAtomicNum() > 1
            and nbr.HasProp("_poly_csp_manifest_source")
            and nbr.GetProp("_poly_csp_manifest_source") != "backbone"
            for nbr in atom.GetNeighbors()
        ):
            out.setdefault(residue_index, set()).add(
                _SUBSTITUTED_SITE_ALLOWED_MISSING[atom_name]
            )
    return out


def map_backbone_to_glycam(
    mol: Chem.Mol,
    glycam_params: GlycamParams,
) -> GlycamMappingResult:
    """Map the backbone subset of a forcefield-domain molecule onto GLYCAM names."""
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "GLYCAM mapping requires a forcefield-domain molecule from build_forcefield_molecule()."
        )

    polymer = _require_mol_prop(mol, "_poly_csp_polymer").strip().lower()
    representation = _require_mol_prop(mol, "_poly_csp_representation").strip().lower()
    end_mode = _require_mol_prop(mol, "_poly_csp_end_mode").strip().lower()
    if polymer != glycam_params.polymer:
        raise ValueError(
            f"GLYCAM params were loaded for polymer {glycam_params.polymer!r}, got {polymer!r}."
        )
    if representation != glycam_params.representation:
        raise ValueError(
            "Phase 2 GLYCAM mapping currently supports only molecules whose "
            f"representation matches {glycam_params.representation!r}; got {representation!r}."
        )
    if end_mode != glycam_params.end_mode:
        raise ValueError(
            "Phase 2 GLYCAM mapping currently supports only molecules whose end mode "
            f"matches {glycam_params.end_mode!r}; got {end_mode!r}."
        )

    if not mol.HasProp("_poly_csp_dp"):
        raise ValueError("Forcefield-domain molecule is missing _poly_csp_dp.")
    dp = int(mol.GetIntProp("_poly_csp_dp"))
    residue_roles = glycam_residue_roles_for_dp(dp)
    allowed_missing = _allowed_missing_hydrogens_by_residue(mol)

    observed_glycam_names: dict[int, set[str]] = {}
    assignments: list[GlycamAtomAssignment] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_manifest_source"):
            raise ValueError(
                f"Atom {atom.GetIdx()} is missing _poly_csp_manifest_source; expected forcefield manifest metadata."
            )
        if atom.GetProp("_poly_csp_manifest_source") != "backbone":
            continue
        if not atom.HasProp("_poly_csp_atom_name"):
            raise ValueError(
                f"Atom {atom.GetIdx()} is missing _poly_csp_atom_name; expected forcefield manifest metadata."
            )

        parent = _parent_backbone_atom(mol, atom)
        if not parent.HasProp("_poly_csp_residue_index"):
            raise ValueError(
                f"Backbone parent atom {parent.GetIdx()} is missing _poly_csp_residue_index."
            )
        residue_index = int(parent.GetIntProp("_poly_csp_residue_index"))
        if residue_index < 0 or residue_index >= len(residue_roles):
            raise ValueError(
                f"Backbone atom {atom.GetIdx()} refers to invalid residue index {residue_index}."
            )

        residue_role = residue_roles[residue_index]
        residue_template = glycam_params.residue_templates.get(residue_role)
        if residue_template is None:
            raise ValueError(
                "No GLYCAM residue template is available for supported residue role "
                f"{residue_role!r}."
            )

        generic_name = atom.GetProp("_poly_csp_atom_name")
        glycam_name = _generic_to_glycam_name(generic_name)
        if (residue_role, glycam_name) not in glycam_params.atom_params:
            raise ValueError(
                f"Residue {residue_index} ({residue_role}) atom {generic_name!r} does not "
                f"map to a supported GLYCAM atom identity {glycam_name!r}."
            )

        observed_glycam_names.setdefault(residue_index, set()).add(glycam_name)
        assignments.append(
            GlycamAtomAssignment(
                atom_index=int(atom.GetIdx()),
                residue_index=residue_index,
                residue_role=residue_role,
                glycam_residue_name=residue_template.residue_name,
                generic_atom_name=generic_name,
                glycam_atom_name=glycam_name,
            )
        )

    for residue_index, residue_role in enumerate(residue_roles):
        residue_template = glycam_params.residue_templates[residue_role]
        observed = observed_glycam_names.get(residue_index, set())
        expected = set(residue_template.atom_names)
        if observed != expected:
            missing_set = expected.difference(observed)
            extra = sorted(observed.difference(expected))
            residue_allowed_missing = allowed_missing.get(residue_index, set())
            if missing_set.issubset(residue_allowed_missing) and not extra:
                continue
            missing = sorted(missing_set)
            raise ValueError(
                "Backbone residue does not match the supported GLYCAM atom set for "
                f"residue {residue_index} ({residue_role}). Missing={missing}, extra={extra}."
            )

    assignments.sort(key=lambda item: item.atom_index)
    return GlycamMappingResult(
        polymer=polymer,
        representation=representation,
        end_mode=end_mode,
        assignments=tuple(assignments),
    )
