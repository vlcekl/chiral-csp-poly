from __future__ import annotations

import json

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.topology.terminals import apply_terminal_mode


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        repeat_residues=4,
        repeat_turns=3,
        axial_repeat_A=14.8,
        handedness="left",
    )


def test_build_forcefield_molecule_assigns_manifest_names_and_pdb_info() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = build_backbone_structure(topology, _helix()).mol

    result = build_forcefield_molecule(mol)

    assert len(result.manifest) == result.mol.GetNumAtoms()
    assert result.mol.GetIntProp("_poly_csp_manifest_schema_version") == 1
    assert all(atom.GetPDBResidueInfo() is not None for atom in result.mol.GetAtoms())

    names = {atom.GetProp("_poly_csp_atom_name") for atom in result.mol.GetAtoms()}
    assert {"C1", "O1", "H1", "HO1", "HO4", "H61", "H62", "HO6"}.issubset(names)


def test_build_forcefield_molecule_preserves_selector_connector_and_cap_identity() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = SelectorRegistry.get("35dmpc")

    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    topology = apply_terminal_mode(
        topology,
        mode="capped",
        caps={"left": "acetyl", "right": "methoxy"},
        representation="natural_oh",
    )
    mol = build_backbone_structure(topology, _helix()).mol
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    result = build_forcefield_molecule(mol)
    maps = json.loads(result.mol.GetProp("_poly_csp_residue_label_map_json"))
    o1 = result.mol.GetAtomWithIdx(maps[0]["O1"])

    selector_entries = [entry for entry in result.manifest if entry.component == "selector"]
    connector_entries = [entry for entry in result.manifest if entry.component == "connector"]
    left_caps = [entry for entry in result.manifest if entry.source == "terminal_cap_left"]
    right_caps = [entry for entry in result.manifest if entry.source == "terminal_cap_right"]

    assert sum(1 for nbr in o1.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert selector_entries
    assert all(entry.atom_name.startswith(("S", "H")) for entry in selector_entries)
    assert connector_entries
    assert any(entry.connector_role == "amide_n" for entry in connector_entries)
    assert left_caps
    assert right_caps


def test_build_forcefield_molecule_rejects_implicit_hydrogen_inputs() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")

    try:
        build_forcefield_molecule(mol)
    except ValueError as exc:
        assert "implicit hydrogens" in str(exc) or "_poly_csp_component" in str(exc)
    else:
        raise AssertionError("Expected build_forcefield_molecule() to reject a heavy-only input.")
