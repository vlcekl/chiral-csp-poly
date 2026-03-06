from __future__ import annotations

import pytest
import openmm as mm
from rdkit import Chem

from poly_csp.forcefield.connectors import (
    CappedMonomerFragment,
    ConnectorParams,
    build_capped_monomer_fragment,
    extract_linkage_params_from_system,
    parameterize_capped_monomer,
)
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.selector_library.tmb import make_tmb_template
from poly_csp.topology.selectors import SelectorTemplate


def _mock_connector_system(fragment: CappedMonomerFragment) -> mm.System:
    system = mm.System()
    for _ in range(fragment.mol.GetNumAtoms()):
        system.addParticle(12.0)

    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    torsion_force = mm.PeriodicTorsionForce()

    connector_roles = set(fragment.connector_atom_roles.values())
    selector_core_roles = sorted(
        role for role in fragment.atom_roles
        if role.startswith("SL_") and role not in connector_roles
    )

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

    system.addForce(bond_force)
    system.addForce(angle_force)
    system.addForce(torsion_force)
    return system


def test_parameterize_capped_monomer_validates_inputs() -> None:
    bad_selector = SelectorTemplate(
        name="bad",
        mol=Chem.Mol(),
        attach_atom_idx=0,
        dihedrals={},
    )

    with pytest.raises(ValueError, match="selector_template"):
        parameterize_capped_monomer("amylose", bad_selector, site="C6")

    with pytest.raises(ValueError, match="site"):
        parameterize_capped_monomer("amylose", make_35_dmpc_template(), site="")


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
    assert len(frag.atom_roles) == frag.mol.GetNumAtoms()


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

    assert isinstance(out, ConnectorParams)
    assert ("BB_O6", frag.connector_atom_roles["carbonyl_c"]) in out.bond_params
    assert ("BB_C6", "BB_O6", frag.connector_atom_roles["carbonyl_c"]) in out.angle_params
    assert len(out.torsion_params) == 1
    assert out.torsion_params[0][0][1] == "BB_O6"
    assert all("BB_C5" not in roles for roles in out.bond_params)
