# tests/test_pdb_residue_info.py
"""Verify PDB output contains residue names and chain IDs."""
from __future__ import annotations

import numpy as np

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.io.pdb import write_pdb_from_rdkit
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorRegistry


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0 * np.pi / 2.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def test_pdb_contains_chain_ids(tmp_path) -> None:
    template = make_glucose_template("amylose")
    selector = SelectorRegistry.get("35dmpc")
    topology = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = build_backbone_structure(topology, _helix()).mol
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    mol = build_forcefield_molecule(mol).mol

    pdb_path = tmp_path / "test.pdb"
    write_pdb_from_rdkit(mol, pdb_path)
    text = pdb_path.read_text(encoding="utf-8")

    atom_lines = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    assert atom_lines

    chains = {line[21] for line in atom_lines if len(line) > 21}
    assert "A" in chains
    assert "B" in chains


def test_pdb_contains_residue_names(tmp_path) -> None:
    template = make_glucose_template("amylose")
    topology = polymerize(template=template, dp=3, linkage="1-4", anomer="alpha")
    mol = build_forcefield_molecule(build_backbone_structure(topology, _helix()).mol).mol

    pdb_path = tmp_path / "test_backbone.pdb"
    write_pdb_from_rdkit(mol, pdb_path)
    text = pdb_path.read_text(encoding="utf-8")

    atom_lines = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert atom_lines

    res_names = {line[17:20].strip() for line in atom_lines if len(line) > 20}
    assert "GLC" in res_names


def test_pdb_uses_forcefield_model_atom_names(tmp_path) -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = SelectorRegistry.get("35dmpc")
    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = build_backbone_structure(topology, _helix()).mol
    mol = attach_selector(
        mol_polymer=mol,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    final_mol = build_forcefield_molecule(mol).mol

    pdb_path = tmp_path / "test_all_atom.pdb"
    write_pdb_from_rdkit(final_mol, pdb_path)
    text = pdb_path.read_text(encoding="utf-8")

    atom_lines = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    atom_names = {line[12:16].strip() for line in atom_lines}
    assert "HO1" in atom_names
    assert "H61" in atom_names
    assert any(name.startswith("S") for name in atom_names)
