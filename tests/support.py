"""Shared test-only helpers around canonical runtime builders."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.connectors import (
    ConnectorAngleTemplate,
    ConnectorAtomParams,
    ConnectorBondTemplate,
    ConnectorParams,
    ConnectorToken,
    ConnectorTorsionTemplate,
)
from poly_csp.forcefield.gaff import (
    SelectorAngleTemplate,
    SelectorAtomParams,
    SelectorBondTemplate,
    SelectorFragmentParams,
    SelectorTorsionTemplate,
)
from poly_csp.forcefield.glycam import (
    GlycamAtomParams,
    GlycamAtomToken,
    GlycamBondTemplate,
    GlycamLinkageTemplate,
    GlycamParams,
    GlycamResidueTemplate,
    glycam_residue_roles_for_dp,
)
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.runtime_params import RuntimeParamCacheSummary, RuntimeParams
from poly_csp.structure.backbone_builder import build_backbone_heavy_coords, build_backbone_structure
from poly_csp.structure.pbc import compute_helical_box_vectors, set_box_vectors
from poly_csp.topology.monomers import GlucoseMonomerTemplate, make_glucose_template
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.topology.terminals import apply_terminal_mode


_GLYCAM_HYDROGEN_ALIASES = {
    "HO1": "H1O",
    "HO2": "H2O",
    "HO3": "H3O",
    "HO4": "H4O",
    "HO6": "H6O",
}


def test_helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-4.71238898038469,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=4.933333333333334,
        handedness="left",
    )


def build_backbone_coords(
    template: GlucoseMonomerTemplate,
    helix: HelixSpec,
    dp: int,
) -> np.ndarray:
    """Return canonical heavy-atom backbone coordinates for tests."""
    return build_backbone_heavy_coords(template, helix, dp)


def assign_conformer(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    """Attach coordinates to a test molecule without any runtime fallback path."""
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3); got {xyz.shape}")
    if xyz.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"Atom count mismatch: coords has {xyz.shape[0]}, mol has {mol.GetNumAtoms()}."
        )

    out = Chem.Mol(mol)
    out.RemoveAllConformers()

    conf = Chem.Conformer(out.GetNumAtoms())
    for atom_idx, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
    out.AddConformer(conf, assignId=True)
    return out


def build_forcefield_mol(
    *,
    polymer: str = "amylose",
    dp: int = 2,
    selector: SelectorTemplate | None = None,
    site: str = "C6",
    end_mode: str = "open",
    end_caps: dict[str, str] | None = None,
) -> Chem.Mol:
    template = make_glucose_template(polymer, monomer_representation="anhydro")
    topology = polymerize(
        template=template,
        dp=dp,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    topology = apply_terminal_mode(
        mol=topology,
        mode=end_mode,  # type: ignore[arg-type]
        caps=dict(end_caps or {}),
        representation="anhydro",
    )
    structure = build_backbone_structure(topology, test_helix()).mol
    if end_mode == "periodic":
        Lx_A, Ly_A, Lz_A = compute_helical_box_vectors(
            structure,
            test_helix(),
            dp=dp,
            padding_A=30.0,
        )
        set_box_vectors(structure, Lx_A, Ly_A, Lz_A)
    if selector is not None:
        for residue_index in range(dp):
            structure = attach_selector(
                mol_polymer=structure,
                residue_index=residue_index,
                site=site,  # type: ignore[arg-type]
                selector=selector,
            )
    return build_forcefield_molecule(structure).mol


def make_fake_runtime_params(
    mol: Chem.Mol,
    *,
    selector: SelectorTemplate | None = None,
    site: str | None = None,
) -> RuntimeParams:
    selector_params_by_name: dict[str, SelectorFragmentParams] = {}
    connector_params_by_key: dict[tuple[str, str], ConnectorParams] = {}
    if selector is not None:
        if site is None:
            raise ValueError("site is required when selector fake params are requested.")
        selector_params_by_name[selector.name] = _fake_selector_params(mol, selector.name)
        connector_params_by_key[(selector.name, site)] = _fake_connector_params(
            mol,
            selector_name=selector.name,
            linkage_type=selector.linkage_type,
            site=site,
        )

    return RuntimeParams(
        glycam=_fake_glycam_params(mol),
        selector_params_by_name=selector_params_by_name,
        connector_params_by_key=connector_params_by_key,
        cache_summary=RuntimeParamCacheSummary(enabled=False, cache_dir=None),
        source_manifest={"runtime": {"cache": {"kind": "test_fake"}}},
    )


def _glycam_name(atom_name: str) -> str:
    return _GLYCAM_HYDROGEN_ALIASES.get(atom_name, atom_name)


def _selector_names(mol: Chem.Mol) -> list[str]:
    return [
        atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "selector"
    ]


def _connector_role_map(mol: Chem.Mol) -> dict[str, str]:
    return {
        atom.GetProp("_poly_csp_connector_role"): atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "connector"
        and atom.HasProp("_poly_csp_connector_role")
    }


def _fake_glycam_params(mol: Chem.Mol) -> GlycamParams:
    polymer = mol.GetProp("_poly_csp_polymer")
    representation = mol.GetProp("_poly_csp_representation")
    end_mode = mol.GetProp("_poly_csp_end_mode")
    dp = int(mol.GetIntProp("_poly_csp_dp"))
    role_end_mode = end_mode if end_mode in {"open", "periodic"} else "open"
    residue_roles = glycam_residue_roles_for_dp(dp, end_mode=role_end_mode)  # type: ignore[arg-type]
    residue_names = {
        "amylose": {
            "terminal_reducing": "4GA",
            "internal": "4GA",
            "terminal_nonreducing": "0GA",
            "periodic": "4GA",
        },
        "cellulose": {
            "terminal_reducing": "4GB",
            "internal": "4GB",
            "terminal_nonreducing": "0GB",
            "periodic": "4GB",
        },
    }[polymer]

    atom_indices_by_residue: dict[int, dict[str, int]] = defaultdict(dict)
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_manifest_source"):
            continue
        if atom.GetProp("_poly_csp_manifest_source") != "backbone":
            continue
        residue_index = int(atom.GetIntProp("_poly_csp_residue_index"))
        atom_name = atom.GetProp("_poly_csp_atom_name")
        atom_indices_by_residue[residue_index][_glycam_name(atom_name)] = int(atom.GetIdx())

    atom_params: dict[tuple[str, str], GlycamAtomParams] = {}
    residue_templates: dict[str, GlycamResidueTemplate] = {}
    residue_bonds: dict[
        str,
        dict[tuple[tuple[int, str], tuple[int, str]], GlycamBondTemplate],
    ] = defaultdict(dict)

    for residue_index, residue_role in enumerate(residue_roles):
        atom_names = tuple(sorted(atom_indices_by_residue[residue_index]))
        residue_templates[residue_role] = GlycamResidueTemplate(
            residue_role=residue_role,
            residue_name=residue_names[residue_role],
            atom_names=atom_names,
            bonds=(),
            angles=(),
            torsions=(),
        )
        for atom_name in atom_names:
            atom_params[(residue_role, atom_name)] = GlycamAtomParams(
                charge_e=0.0,
                sigma_nm=0.3,
                epsilon_kj_per_mol=0.1,
                residue_name=residue_names[residue_role],
                source_atom_name=atom_name,
            )

    linkage_bonds: dict[
        tuple[str, str],
        dict[tuple[tuple[int, str], tuple[int, str]], GlycamBondTemplate],
    ] = defaultdict(dict)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if (
            begin.HasProp("_poly_csp_manifest_source")
            and end.HasProp("_poly_csp_manifest_source")
            and begin.GetProp("_poly_csp_manifest_source") == "backbone"
            and end.GetProp("_poly_csp_manifest_source") == "backbone"
        ):
            res_i = int(begin.GetIntProp("_poly_csp_residue_index"))
            res_j = int(end.GetIntProp("_poly_csp_residue_index"))
            name_i = _glycam_name(begin.GetProp("_poly_csp_atom_name"))
            name_j = _glycam_name(end.GetProp("_poly_csp_atom_name"))
            token_i = GlycamAtomToken(residue_offset=0, atom_name=name_i)
            if res_i == res_j:
                token_j = GlycamAtomToken(residue_offset=0, atom_name=name_j)
                key = (
                    (token_i.residue_offset, token_i.atom_name),
                    (token_j.residue_offset, token_j.atom_name),
                )
                if key[0] > key[1]:
                    key = (key[1], key[0])
                    token_i, token_j = token_j, token_i
                residue_bonds[residue_roles[res_i]][key] = GlycamBondTemplate(
                    atoms=(token_i, token_j),
                    length_nm=0.15,
                    k_kj_per_mol_nm2=100.0,
                )
                continue

            if end_mode == "periodic" and {res_i, res_j} == {0, dp - 1}:
                left_residue = dp - 1
                right_residue = 0
            else:
                left_residue = min(res_i, res_j)
                right_residue = max(res_i, res_j)
            left_token = GlycamAtomToken(
                residue_offset=0,
                atom_name=name_i if res_i == left_residue else name_j,
            )
            right_token = GlycamAtomToken(
                residue_offset=1,
                atom_name=name_j if res_j == right_residue else name_i,
            )
            key = (
                (left_token.residue_offset, left_token.atom_name),
                (right_token.residue_offset, right_token.atom_name),
            )
            linkage_bonds[(residue_roles[left_residue], residue_roles[right_residue])][key] = GlycamBondTemplate(
                atoms=(left_token, right_token),
                length_nm=0.14,
                k_kj_per_mol_nm2=120.0,
            )

    for residue_role, template in list(residue_templates.items()):
        residue_templates[residue_role] = GlycamResidueTemplate(
            residue_role=template.residue_role,
            residue_name=template.residue_name,
            atom_names=template.atom_names,
            bonds=tuple(sorted(residue_bonds[residue_role].values(), key=lambda item: item.atoms)),
            angles=(),
            torsions=(),
        )

    linkage_templates = {
        pair: GlycamLinkageTemplate(
            residue_roles=pair,  # type: ignore[arg-type]
            bonds=tuple(sorted(bonds.values(), key=lambda item: item.atoms)),
            angles=(),
            torsions=(),
        )
        for pair, bonds in linkage_bonds.items()
    }

    return GlycamParams(
        polymer=polymer,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        end_mode=end_mode,  # type: ignore[arg-type]
        atom_params=atom_params,
        residue_templates=residue_templates,
        linkage_templates=linkage_templates,
        supported_states=tuple(
            (polymer, representation, end_mode, role)
            for role in residue_templates
        ),
        provenance={"parameter_backend": "test_fake_glycam"},
    )


def _fake_selector_params(mol: Chem.Mol, selector_name: str) -> SelectorFragmentParams:
    atom_names = _selector_names(mol)
    bonds: tuple[SelectorBondTemplate, ...] = ()
    angles: tuple[SelectorAngleTemplate, ...] = ()
    torsions: tuple[SelectorTorsionTemplate, ...] = ()
    if len(atom_names) >= 2:
        bonds = (
            SelectorBondTemplate(
                atom_names=(atom_names[0], atom_names[1]),
                length_nm=0.145,
                k_kj_per_mol_nm2=220.0,
            ),
        )
    if len(atom_names) >= 3:
        angles = (
            SelectorAngleTemplate(
                atom_names=(atom_names[0], atom_names[1], atom_names[2]),
                theta0_rad=2.09,
                k_kj_per_mol_rad2=60.0,
            ),
        )
    if len(atom_names) >= 4:
        torsions = (
            SelectorTorsionTemplate(
                atom_names=(atom_names[0], atom_names[1], atom_names[2], atom_names[3]),
                periodicity=2,
                phase_rad=3.14,
                k_kj_per_mol=4.0,
            ),
        )
    return SelectorFragmentParams(
        selector_name=selector_name,
        atom_params={
            atom_name: SelectorAtomParams(
                atom_name=atom_name,
                charge_e=-0.05,
                sigma_nm=0.31,
                epsilon_kj_per_mol=0.12,
            )
            for atom_name in atom_names
        },
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        source_prmtop="selector_fragment.prmtop",
        fragment_atom_count=len(atom_names),
    )


def _fake_connector_params(
    mol: Chem.Mol,
    *,
    selector_name: str,
    linkage_type: str,
    site: str,
) -> ConnectorParams:
    connector_names = [
        atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "connector"
    ]
    selector_names = _selector_names(mol)
    role_map = _connector_role_map(mol)
    anchor = f"O{site[1:]}"
    carbon = site
    if linkage_type == "carbamate":
        torsions = (
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("backbone", carbon),
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("connector", role_map["amide_n"]),
                ),
                periodicity=2,
                phase_rad=3.14,
                k_kj_per_mol=8.0,
            ),
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("connector", role_map["amide_n"]),
                    ConnectorToken("selector", selector_names[0]),
                ),
                periodicity=2,
                phase_rad=0.0,
                k_kj_per_mol=7.0,
            ),
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("connector", role_map["carbonyl_o"]),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("connector", role_map["amide_n"]),
                    ConnectorToken("selector", selector_names[1]),
                ),
                periodicity=2,
                phase_rad=0.0,
                k_kj_per_mol=6.0,
            ),
        )
    else:
        torsions = (
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("connector", role_map["carbonyl_o"]),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("selector", selector_names[0]),
                    ConnectorToken("selector", selector_names[1]),
                ),
                periodicity=2,
                phase_rad=0.0,
                k_kj_per_mol=6.0,
            ),
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("backbone", carbon),
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("selector", selector_names[0]),
                ),
                periodicity=2,
                phase_rad=3.14,
                k_kj_per_mol=5.0,
            ),
        )
    return ConnectorParams(
        polymer=mol.GetProp("_poly_csp_polymer"),  # type: ignore[arg-type]
        selector_name=selector_name,
        site=site,  # type: ignore[arg-type]
        monomer_representation="natural_oh",
        linkage_type=linkage_type,  # type: ignore[arg-type]
        atom_params={
            atom_name: ConnectorAtomParams(
                atom_name=atom_name,
                charge_e=0.04,
                sigma_nm=0.29,
                epsilon_kj_per_mol=0.09,
            )
            for atom_name in connector_names
        },
        connector_role_atom_names=role_map,
        bonds=(
            ConnectorBondTemplate(
                atoms=(
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                ),
                length_nm=0.136,
                k_kj_per_mol_nm2=640.0,
            ),
        ),
        angles=(
            ConnectorAngleTemplate(
                atoms=(
                    ConnectorToken("backbone", carbon),
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                ),
                theta0_rad=2.04,
                k_kj_per_mol_rad2=77.0,
            ),
        ),
        torsions=torsions,
        source_prmtop="connector_fragment.prmtop",
        fragment_atom_count=len(connector_names),
    )
