from __future__ import annotations

import openmm as mm
import pytest
from rdkit import Chem

from poly_csp.forcefield.connectors import (
    CappedMonomerFragment,
    ConnectorParams,
    ConnectorToken,
    build_capped_monomer_fragment,
    extract_linkage_params_from_system,
    load_connector_params,
)
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.structure.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.structure.selector_library.tmb import make_tmb_template
from poly_csp.topology.selectors import SelectorTemplate


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
