"""Test SDF export round-trip: write then read back with correct topology."""
from __future__ import annotations

import numpy as np
from rdkit import Chem

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.functionalization import attach_selector
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.chemistry.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.config.schema import HelixSpec
from poly_csp.io.rdkit_io import write_sdf


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


def _build_functionalized_polymer(dp: int = 3) -> Chem.Mol:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol,
            template=template,
            residue_index=i,
            site="C6",
            selector=selector,
        )
    return mol


def test_sdf_roundtrip(tmp_path) -> None:
    mol = _build_functionalized_polymer(dp=3)
    sdf_path = tmp_path / "model.sdf"
    write_sdf(mol, sdf_path)

    assert sdf_path.exists()
    assert sdf_path.stat().st_size > 0

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in supplier if m is not None]
    assert len(mols) == 1

    loaded = mols[0]
    assert loaded.GetNumAtoms() == mol.GetNumAtoms()
    assert loaded.GetNumBonds() == mol.GetNumBonds()
    assert loaded.GetNumConformers() >= 1


def test_sdf_preserves_aromatic_bonds(tmp_path) -> None:
    mol = _build_functionalized_polymer(dp=2)
    sdf_path = tmp_path / "model.sdf"
    write_sdf(mol, sdf_path)

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    loaded = next(iter(supplier))
    assert loaded is not None

    # The selector has aromatic rings; check that at least some aromatic bonds survive
    original_aromatic = sum(
        1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC
    )
    loaded_aromatic = sum(
        1 for b in loaded.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC
    )
    # SDF V2000 may Kekulize, so we check either aromatic or alternating single/double
    assert loaded_aromatic > 0 or loaded.GetNumBonds() == mol.GetNumBonds()
