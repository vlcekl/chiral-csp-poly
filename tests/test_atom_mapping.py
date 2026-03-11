from __future__ import annotations

from poly_csp.config.schema import HelixSpec
from tests.support import build_backbone_coords
from poly_csp.topology.atom_mapping import (
    attachment_instance_maps,
    backbone_indices,
    build_atom_map,
    connector_instance_maps,
    connector_indices,
    selector_indices,
    selector_instance_maps,
)
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorRegistry


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        repeat_residues=4,
        repeat_turns=3,
        axial_repeat_A=14.8,
        handedness="left",
    )


def test_atom_mapping_sets_are_disjoint_and_exhaustive() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    mol = polymerize(template=template, dp=4, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, build_backbone_coords(template, _helix(), dp=4))
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=1,
        site="C6",
        selector=selector,
    )

    atom_map = build_atom_map(mol)
    bb = backbone_indices(mol)
    sel = selector_indices(mol)
    conn = connector_indices(mol)

    assert len(atom_map) == mol.GetNumAtoms()
    assert bb.isdisjoint(sel)
    assert bb.isdisjoint(conn)
    assert sel.isdisjoint(conn)
    assert len(bb | sel | conn) == mol.GetNumAtoms()
    assert len(sel) > 0
    assert len(conn) == len(selector.connector_local_roles) + 1


def test_selector_instance_maps_present_after_attachment() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, build_backbone_coords(template, _helix(), dp=2))
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    mappings = selector_instance_maps(mol)
    assert mappings
    instance_map = next(iter(mappings.values()))
    assert len(instance_map) > 0


def test_attachment_instance_maps_include_selector_and_connector_atoms() -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, build_backbone_coords(template, _helix(), dp=2))
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    attached = attachment_instance_maps(mol)
    selector_only = selector_instance_maps(mol)
    connector_only = connector_instance_maps(mol)

    assert attached
    instance_id = next(iter(attached))
    assert len(attached[instance_id]) == selector.mol.GetNumAtoms() - 1
    assert len(selector_only[instance_id]) + len(connector_only[instance_id]) == len(
        attached[instance_id]
    )
    assert set(selector.connector_local_roles).issubset(set(connector_only[instance_id]))
    assert len(connector_only[instance_id]) == len(selector.connector_local_roles) + 1
