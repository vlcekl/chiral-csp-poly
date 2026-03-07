from __future__ import annotations

from dataclasses import replace
import os
import shutil

import openmm as mm
import pytest
from rdkit import Chem

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.connectors import (
    CappedMonomerFragment,
    ConnectorAngleTemplate,
    ConnectorAtomParams,
    ConnectorBondTemplate,
    ConnectorParams,
    ConnectorToken,
    ConnectorTorsionTemplate,
    build_capped_monomer_fragment,
    extract_linkage_params_from_system,
    load_connector_params,
    validate_connector_params,
)
from poly_csp.forcefield.gaff import SelectorAtomParams, SelectorFragmentParams
from poly_csp.forcefield.glycam import (
    GlycamAtomParams,
    GlycamParams,
    GlycamResidueTemplate,
)
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.system_builder import create_system
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.structure.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.structure.selector_library.tmb import make_tmb_template
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.topology.terminals import apply_terminal_mode


_HELIX = {
    "name": "connector_test_helix",
    "theta_rad": -4.71238898038469,
    "rise_A": 3.7,
    "repeat_residues": 4,
    "repeat_turns": 3,
    "residues_per_turn": 4.0 / 3.0,
    "pitch_A": 4.933333333333334,
    "handedness": "left",
}
_GLYCAM_HYDROGEN_ALIASES = {
    "HO1": "H1O",
    "HO2": "H2O",
    "HO3": "H3O",
    "HO4": "H4O",
    "HO6": "H6O",
}


def _forcefield_atom_names_by_fragment_role(fragment: CappedMonomerFragment) -> dict[str, str]:
    forcefield = build_forcefield_molecule(fragment.mol).mol
    out: dict[str, str] = {}
    for atom in forcefield.GetAtoms():
        if not atom.HasProp("_poly_csp_fragment_role"):
            continue
        out[atom.GetProp("_poly_csp_fragment_role")] = atom.GetProp("_poly_csp_atom_name")
    return out


def _mock_connector_system(fragment: CappedMonomerFragment) -> mm.System:
    system = mm.System()
    for atom_idx, _ in enumerate(fragment.mol.GetAtoms()):
        system.addParticle(12.0)

    nonbonded = mm.NonbondedForce()
    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    torsion_force = mm.PeriodicTorsionForce()

    connector_roles = set(fragment.connector_atom_roles.values())
    selector_core_roles = sorted(
        role
        for role in fragment.atom_roles
        if role.startswith("SL_") and role not in connector_roles
    )

    for atom_idx in range(fragment.mol.GetNumAtoms()):
        charge = 0.0
        sigma = 0.25
        epsilon = 0.1
        if atom_idx in fragment.connector_roles.values():
            charge = -0.3
            sigma = 0.32
            epsilon = 0.2
        nonbonded.addParticle(charge, sigma, epsilon)

    bond_force.addBond(
        fragment.atom_roles["BB_C6"],
        fragment.atom_roles["BB_O6"],
        0.15,
        111.0,
    )
    bond_force.addBond(
        fragment.atom_roles["BB_O6"],
        fragment.atom_roles[fragment.connector_atom_roles["carbonyl_c"]],
        0.136,
        654.0,
    )
    bond_force.addBond(
        fragment.atom_roles[selector_core_roles[0]],
        fragment.atom_roles[selector_core_roles[1]],
        0.145,
        222.0,
    )

    angle_force.addAngle(
        fragment.atom_roles["BB_C6"],
        fragment.atom_roles["BB_O6"],
        fragment.atom_roles[fragment.connector_atom_roles["carbonyl_c"]],
        2.04,
        77.0,
    )
    angle_force.addAngle(
        fragment.atom_roles["BB_C5"],
        fragment.atom_roles["BB_C6"],
        fragment.atom_roles["BB_O6"],
        1.91,
        88.0,
    )

    torsion_force.addTorsion(
        fragment.atom_roles["BB_C6"],
        fragment.atom_roles["BB_O6"],
        fragment.atom_roles[fragment.connector_atom_roles["carbonyl_c"]],
        fragment.atom_roles[fragment.connector_atom_roles["amide_n"]],
        2,
        3.14,
        9.5,
    )
    torsion_force.addTorsion(
        fragment.atom_roles["BB_O6"],
        fragment.atom_roles[fragment.connector_atom_roles["carbonyl_c"]],
        fragment.atom_roles[fragment.connector_atom_roles["amide_n"]],
        fragment.atom_roles[selector_core_roles[0]],
        2,
        3.14,
        11.5,
    )
    torsion_force.addTorsion(
        fragment.atom_roles[fragment.connector_atom_roles["carbonyl_o"]],
        fragment.atom_roles[fragment.connector_atom_roles["carbonyl_c"]],
        fragment.atom_roles[fragment.connector_atom_roles["amide_n"]],
        fragment.atom_roles[selector_core_roles[0]],
        2,
        0.0,
        8.5,
    )
    torsion_force.addTorsion(
        fragment.atom_roles[selector_core_roles[0]],
        fragment.atom_roles[selector_core_roles[1]],
        fragment.atom_roles[selector_core_roles[2]],
        fragment.atom_roles[selector_core_roles[3]],
        3,
        0.0,
        12.0,
    )

    system.addForce(nonbonded)
    system.addForce(bond_force)
    system.addForce(angle_force)
    system.addForce(torsion_force)
    return system


def _forcefield_selector_mol(
    selector: SelectorTemplate,
    *,
    polymer: str = "amylose",
    site: str = "C6",
):
    template = make_glucose_template(polymer, monomer_representation="anhydro")
    topology = polymerize(
        template=template,
        dp=1,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    topology = apply_terminal_mode(
        mol=topology,
        mode="open",
        caps={},
        representation="anhydro",
    )
    structure = build_backbone_structure(
        topology,
        helix_spec=HelixSpec(**_HELIX),
    ).mol
    structure = attach_selector(
        mol_polymer=structure,
        residue_index=0,
        site=site,
        selector=selector,
    )
    return build_forcefield_molecule(structure).mol


def _atom_names_by_source(mol: Chem.Mol, source: str) -> list[str]:
    return sorted(
        atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == source
        and atom.HasProp("_poly_csp_atom_name")
    )


def _glycam_name(atom_name: str) -> str:
    return _GLYCAM_HYDROGEN_ALIASES.get(atom_name, atom_name)


def _fake_glycam_params(mol: Chem.Mol) -> GlycamParams:
    residue_role = "terminal_nonreducing"
    residue_name = "0GA" if mol.GetProp("_poly_csp_polymer") == "amylose" else "0GB"
    glycam_names = tuple(
        sorted(
            {
                _glycam_name(atom.GetProp("_poly_csp_atom_name"))
                for atom in mol.GetAtoms()
                if atom.HasProp("_poly_csp_manifest_source")
                and atom.GetProp("_poly_csp_manifest_source") == "backbone"
                and atom.HasProp("_poly_csp_atom_name")
            }
        )
    )
    atom_params = {
        (residue_role, atom_name): GlycamAtomParams(
            charge_e=0.0,
            sigma_nm=0.3,
            epsilon_kj_per_mol=0.1,
            residue_name=residue_name,
            source_atom_name=atom_name,
        )
        for atom_name in glycam_names
    }
    residue_template = GlycamResidueTemplate(
        residue_role=residue_role,
        residue_name=residue_name,
        atom_names=glycam_names,
        bonds=(),
        angles=(),
        torsions=(),
    )
    return GlycamParams(
        polymer=mol.GetProp("_poly_csp_polymer"),
        representation=mol.GetProp("_poly_csp_representation"),
        end_mode=mol.GetProp("_poly_csp_end_mode"),
        atom_params=atom_params,
        residue_templates={residue_role: residue_template},
        linkage_templates={},
        supported_states=(
            (
                mol.GetProp("_poly_csp_polymer"),
                mol.GetProp("_poly_csp_representation"),
                mol.GetProp("_poly_csp_end_mode"),
                residue_role,
            ),
        ),
        provenance={"parameter_backend": "test_fake_glycam"},
    )


def _fake_selector_params(mol: Chem.Mol, selector: SelectorTemplate) -> SelectorFragmentParams:
    return SelectorFragmentParams(
        selector_name=selector.name,
        atom_params={
            atom_name: SelectorAtomParams(
                atom_name=atom_name,
                charge_e=-0.05,
                sigma_nm=0.31,
                epsilon_kj_per_mol=0.12,
            )
            for atom_name in _atom_names_by_source(mol, "selector")
        },
        bonds=(),
        angles=(),
        torsions=(),
        source_prmtop="selector_fragment.prmtop",
        fragment_atom_count=len(_atom_names_by_source(mol, "selector")),
    )


def _fake_connector_params(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    *,
    site: str,
) -> ConnectorParams:
    connector_names = _atom_names_by_source(mol, "connector")
    selector_names = _atom_names_by_source(mol, "selector")
    if len(selector_names) < 2:
        raise AssertionError("Expected at least two selector-core atoms in the forcefield molecule.")
    role_map = {
        atom.GetProp("_poly_csp_connector_role"): atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "connector"
        and atom.HasProp("_poly_csp_connector_role")
        and atom.HasProp("_poly_csp_atom_name")
    }
    anchor = f"O{site[1:]}"
    carbon = site
    torsions: tuple[ConnectorTorsionTemplate, ...] = ()
    if selector.linkage_type == "carbamate":
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
    elif selector.linkage_type == "ester":
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
        polymer=mol.GetProp("_poly_csp_polymer"),
        selector_name=selector.name,
        site=site,
        monomer_representation="natural_oh",
        linkage_type=selector.linkage_type,
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


def test_load_connector_params_validates_inputs() -> None:
    bad_selector = SelectorTemplate(
        name="bad",
        mol=Chem.Mol(),
        attach_atom_idx=0,
        dihedrals={},
    )

    with pytest.raises(ValueError, match="selector_template"):
        load_connector_params("amylose", bad_selector, site="C6")

    with pytest.raises(ValueError, match="site"):
        load_connector_params("amylose", make_35_dmpc_template(), site="")


def test_build_capped_monomer_fragment_assigns_backbone_and_selector_roles() -> None:
    frag = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_35_dmpc_template(),
        site="C6",
    )

    assert isinstance(frag, CappedMonomerFragment)
    assert frag.mol.GetNumConformers() == 1
    assert "BB_O6" in frag.atom_roles
    assert "BB_C6" in frag.atom_roles
    assert any(role.startswith("SL_") for role in frag.atom_roles)
    assert set(frag.connector_roles) == {"carbonyl_c", "carbonyl_o", "amide_n"}


@pytest.mark.parametrize("polymer", ["amylose", "cellulose"])
@pytest.mark.parametrize("site, anchor", [("C2", "BB_O2"), ("C3", "BB_O3"), ("C6", "BB_O6")])
@pytest.mark.parametrize(
    "selector_factory, expected_roles",
    [
        (make_35_dmpc_template, {"carbonyl_c", "carbonyl_o", "amide_n"}),
        (make_tmb_template, {"carbonyl_c", "carbonyl_o"}),
    ],
)
def test_build_capped_monomer_fragment_covers_supported_sites_and_polymers(
    polymer: str,
    site: str,
    anchor: str,
    selector_factory,
    expected_roles: set[str],
) -> None:
    frag = build_capped_monomer_fragment(
        polymer=polymer,
        selector_template=selector_factory(),
        site=site,
    )

    assert anchor in frag.atom_roles
    assert set(frag.connector_roles) == expected_roles
    anchor_idx = frag.atom_roles[anchor]
    connector_indices = set(frag.connector_roles.values())
    assert any(
        int(neighbor.GetIdx()) in connector_indices
        for neighbor in frag.mol.GetAtomWithIdx(anchor_idx).GetNeighbors()
    )


def test_build_capped_monomer_fragment_handles_ester_selector() -> None:
    frag = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_tmb_template(),
        site="C6",
    )

    assert set(frag.connector_roles) == {"carbonyl_c", "carbonyl_o"}
    for atom_idx in frag.connector_roles.values():
        atom = frag.mol.GetAtomWithIdx(atom_idx)
        assert atom.GetProp("_poly_csp_component") == "connector"
        assert atom.HasProp("_poly_csp_fragment_role")


def test_extract_linkage_params_from_system_keeps_only_connector_terms() -> None:
    frag = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_35_dmpc_template(),
        site="C6",
    )
    ref_system = _mock_connector_system(frag)
    out = extract_linkage_params_from_system(ref_system=ref_system, fragment=frag)
    role_names = _forcefield_atom_names_by_fragment_role(frag)

    carbonyl_name = role_names[frag.connector_atom_roles["carbonyl_c"]]
    amide_name = role_names[frag.connector_atom_roles["amide_n"]]

    assert isinstance(out, ConnectorParams)
    assert carbonyl_name in out.atom_params
    assert out.atom_params[carbonyl_name].sigma_nm > 0.0
    assert ConnectorToken("connector", carbonyl_name) in {
        token
        for template in out.bonds
        for token in template.atoms
    }
    assert any(
        template.atoms
        == (
            ConnectorToken("backbone", "O6"),
            ConnectorToken("connector", carbonyl_name),
        )
        or template.atoms
        == (
            ConnectorToken("connector", carbonyl_name),
            ConnectorToken("backbone", "O6"),
        )
        for template in out.bonds
    )
    assert any(
        template.atoms
        == (
            ConnectorToken("backbone", "C6"),
            ConnectorToken("backbone", "O6"),
            ConnectorToken("connector", carbonyl_name),
        )
        or template.atoms
        == (
            ConnectorToken("connector", carbonyl_name),
            ConnectorToken("backbone", "O6"),
            ConnectorToken("backbone", "C6"),
        )
        for template in out.angles
    )
    assert any(
        template.atoms
        == (
            ConnectorToken("backbone", "C6"),
            ConnectorToken("backbone", "O6"),
            ConnectorToken("connector", carbonyl_name),
            ConnectorToken("connector", amide_name),
        )
        for template in out.torsions
    )
    assert out.linkage_type == "carbamate"
    assert set(out.connector_role_atom_names) == {"carbonyl_c", "carbonyl_o", "amide_n"}


def test_validate_connector_params_rejects_terms_without_connector_atoms() -> None:
    frag = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_35_dmpc_template(),
        site="C6",
    )
    ref_system = _mock_connector_system(frag)
    out = extract_linkage_params_from_system(ref_system=ref_system, fragment=frag)

    if not out.bonds:
        pytest.skip("Mock connector payload unexpectedly contains no bonds.")
    bad = ConnectorParams(
        polymer=out.polymer,
        selector_name=out.selector_name,
        site=out.site,
        monomer_representation=out.monomer_representation,
        linkage_type=out.linkage_type,
        atom_params=dict(out.atom_params),
        connector_role_atom_names=dict(out.connector_role_atom_names),
        bonds=(
            ConnectorBondTemplate(
                atoms=(
                    ConnectorToken("selector", "S004"),
                    ConnectorToken("selector", "S005"),
                ),
                length_nm=out.bonds[0].length_nm,
                k_kj_per_mol_nm2=out.bonds[0].k_kj_per_mol_nm2,
            ),
        ),
        angles=out.angles,
        torsions=out.torsions,
        source_prmtop=out.source_prmtop,
        fragment_atom_count=out.fragment_atom_count,
    )

    with pytest.raises(ValueError, match="no connector atoms"):
        validate_connector_params(bad)


def test_validate_connector_params_accepts_ester_planarity_terms() -> None:
    params = ConnectorParams(
        polymer="amylose",
        selector_name="tmb",
        site="C6",
        monomer_representation="natural_oh",
        linkage_type="ester",
        atom_params={
            "C001": ConnectorAtomParams(
                atom_name="C001",
                charge_e=0.12,
                sigma_nm=0.29,
                epsilon_kj_per_mol=0.08,
            ),
            "O001": ConnectorAtomParams(
                atom_name="O001",
                charge_e=-0.22,
                sigma_nm=0.28,
                epsilon_kj_per_mol=0.10,
            ),
        },
        connector_role_atom_names={
            "carbonyl_c": "C001",
            "carbonyl_o": "O001",
        },
        bonds=(
            ConnectorBondTemplate(
                atoms=(
                    ConnectorToken("backbone", "O6"),
                    ConnectorToken("connector", "C001"),
                ),
                length_nm=0.136,
                k_kj_per_mol_nm2=640.0,
            ),
        ),
        angles=(
            ConnectorAngleTemplate(
                atoms=(
                    ConnectorToken("backbone", "C6"),
                    ConnectorToken("backbone", "O6"),
                    ConnectorToken("connector", "C001"),
                ),
                theta0_rad=2.04,
                k_kj_per_mol_rad2=77.0,
            ),
        ),
        torsions=(
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("connector", "O001"),
                    ConnectorToken("connector", "C001"),
                    ConnectorToken("selector", "S001"),
                    ConnectorToken("selector", "S002"),
                ),
                periodicity=2,
                phase_rad=0.0,
                k_kj_per_mol=6.0,
            ),
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("backbone", "C6"),
                    ConnectorToken("backbone", "O6"),
                    ConnectorToken("connector", "C001"),
                    ConnectorToken("selector", "S001"),
                ),
                periodicity=2,
                phase_rad=3.14,
                k_kj_per_mol=5.0,
            ),
        ),
    )

    validate_connector_params(params)


@pytest.mark.parametrize("nonbonded_mode", ["soft", "full"])
def test_create_system_materializes_connector_terms_in_both_runtime_modes(
    nonbonded_mode: str,
) -> None:
    selector = make_35_dmpc_template()
    mol = _forcefield_selector_mol(selector, site="C6")
    glycam = _fake_glycam_params(mol)
    selector_params = _fake_selector_params(mol, selector)
    connector_params = _fake_connector_params(mol, selector, site="C6")

    result = create_system(
        mol,
        glycam_params=glycam,
        selector_params_by_name={selector.name: selector_params},
        connector_params_by_key={(selector.name, "C6"): connector_params},
        nonbonded_mode=nonbonded_mode,
    )

    role_map = connector_params.connector_role_atom_names
    atom_index_by_name = {
        atom.GetProp("_poly_csp_atom_name"): int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_atom_name")
    }
    anchor_pair = {
        atom_index_by_name["O6"],
        atom_index_by_name[role_map["carbonyl_c"]],
    }
    connector_torsion = (
        atom_index_by_name["O6"],
        atom_index_by_name[role_map["carbonyl_c"]],
        atom_index_by_name[role_map["amide_n"]],
        atom_index_by_name[_atom_names_by_source(mol, "selector")[0]],
    )

    bond_force = next(
        force
        for force in (result.system.getForce(i) for i in range(result.system.getNumForces()))
        if isinstance(force, mm.HarmonicBondForce)
    )
    torsion_force = next(
        force
        for force in (result.system.getForce(i) for i in range(result.system.getNumForces()))
        if isinstance(force, mm.PeriodicTorsionForce)
    )

    assert any(
        {int(a), int(b)} == anchor_pair
        for a, b, _, _ in (bond_force.getBondParameters(i) for i in range(bond_force.getNumBonds()))
    )
    assert any(
        tuple(int(value) for value in torsion_force.getTorsionParameters(i)[:4]) == connector_torsion
        for i in range(torsion_force.getNumTorsions())
    )
    assert result.source_manifest["connector"][f"{selector.name}:C6"]["linkage_type"] == "carbamate"
    assert result.source_manifest["connector"][f"{selector.name}:C6"]["connector_role_atom_names"] == role_map

    forces = [result.system.getForce(i) for i in range(result.system.getNumForces())]
    if nonbonded_mode == "soft":
        assert any(isinstance(force, mm.CustomNonbondedForce) for force in forces)
        assert not any(isinstance(force, mm.NonbondedForce) for force in forces)
    else:
        assert any(isinstance(force, mm.NonbondedForce) for force in forces)


def test_create_system_rejects_connector_payload_missing_runtime_atom_params() -> None:
    selector = make_35_dmpc_template()
    mol = _forcefield_selector_mol(selector, site="C6")
    glycam = _fake_glycam_params(mol)
    selector_params = _fake_selector_params(mol, selector)
    connector_params = _fake_connector_params(mol, selector, site="C6")
    extra_connector_names = sorted(
        set(connector_params.atom_params).difference(connector_params.connector_role_atom_names.values())
    )
    assert extra_connector_names, "Expected a connector-only hydrogen outside the role map."
    bad_connector = replace(
        connector_params,
        atom_params={
            name: payload
            for name, payload in connector_params.atom_params.items()
            if name != extra_connector_names[0]
        },
    )

    with pytest.raises(ValueError, match="Connector instance atom-set mismatch"):
        create_system(
            mol,
            glycam_params=glycam,
            selector_params_by_name={selector.name: selector_params},
            connector_params_by_key={(selector.name, "C6"): bad_connector},
            nonbonded_mode="full",
        )


def test_create_system_rejects_connector_terms_for_atoms_outside_the_instance() -> None:
    selector = make_tmb_template()
    mol = _forcefield_selector_mol(selector, site="C6")
    glycam = _fake_glycam_params(mol)
    selector_params = _fake_selector_params(mol, selector)
    connector_params = _fake_connector_params(mol, selector, site="C6")
    assert connector_params.torsions
    bad_torsion = ConnectorTorsionTemplate(
        atoms=(
            connector_params.torsions[0].atoms[0],
            connector_params.torsions[0].atoms[1],
            connector_params.torsions[0].atoms[2],
            ConnectorToken("selector", "S999"),
        ),
        periodicity=connector_params.torsions[0].periodicity,
        phase_rad=connector_params.torsions[0].phase_rad,
        k_kj_per_mol=connector_params.torsions[0].k_kj_per_mol,
    )
    bad_connector = replace(
        connector_params,
        torsions=(bad_torsion, *connector_params.torsions[1:]),
    )

    with pytest.raises(ValueError, match="Selector atom 'S999' is missing from instance"):
        create_system(
            mol,
            glycam_params=glycam,
            selector_params_by_name={selector.name: selector_params},
            connector_params_by_key={(selector.name, "C6"): bad_connector},
            nonbonded_mode="soft",
        )


@pytest.mark.integration
@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("antechamber", "parmchk2", "tleap")),
    reason="AmberTools fragment tools are not available",
)
@pytest.mark.skipif(
    os.environ.get("POLYCSP_RUN_SLOW") != "1",
    reason="set POLYCSP_RUN_SLOW=1 to run slow connector parameterization checks",
)
def test_load_connector_params_extracts_real_carbamate_planarity_terms(tmp_path) -> None:
    params = load_connector_params(
        polymer="amylose",
        selector_template=make_35_dmpc_template(),
        site="C6",
        work_dir=tmp_path / "carbamate_connector",
    )

    assert params.linkage_type == "carbamate"
    assert set(params.connector_role_atom_names) == {"carbonyl_c", "carbonyl_o", "amide_n"}
    carbonyl_c = params.connector_role_atom_names["carbonyl_c"]
    carbonyl_o = params.connector_role_atom_names["carbonyl_o"]
    amide_n = params.connector_role_atom_names["amide_n"]
    assert any(
        carbonyl_c in {token.atom_name for token in template.atoms if token.source == "connector"}
        and amide_n in {token.atom_name for token in template.atoms if token.source == "connector"}
        and any(token.source == "selector" for token in template.atoms)
        for template in params.torsions
    )
    assert any(
        carbonyl_o in {token.atom_name for token in template.atoms if token.source == "connector"}
        and carbonyl_c in {token.atom_name for token in template.atoms if token.source == "connector"}
        for template in params.torsions
    )


@pytest.mark.integration
@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("antechamber", "parmchk2", "tleap")),
    reason="AmberTools fragment tools are not available",
)
@pytest.mark.skipif(
    os.environ.get("POLYCSP_RUN_SLOW") != "1",
    reason="set POLYCSP_RUN_SLOW=1 to run slow connector parameterization checks",
)
def test_load_connector_params_extracts_real_ester_planarity_terms(tmp_path) -> None:
    params = load_connector_params(
        polymer="amylose",
        selector_template=make_tmb_template(),
        site="C6",
        work_dir=tmp_path / "ester_connector",
    )

    assert params.linkage_type == "ester"
    assert set(params.connector_role_atom_names) == {"carbonyl_c", "carbonyl_o"}
    carbonyl_c = params.connector_role_atom_names["carbonyl_c"]
    carbonyl_o = params.connector_role_atom_names["carbonyl_o"]
    assert any(
        carbonyl_o in {token.atom_name for token in template.atoms if token.source == "connector"}
        and carbonyl_c in {token.atom_name for token in template.atoms if token.source == "connector"}
        and any(token.source == "selector" for token in template.atoms)
        for template in params.torsions
    )
    assert any(
        carbonyl_c in {token.atom_name for token in template.atoms if token.source == "connector"}
        and any(token.source == "backbone" and token.atom_name == "O6" for token in template.atoms)
        and any(token.source == "selector" for token in template.atoms)
        for template in params.torsions
    )
