# tests/test_glycam_assembly.py
"""Unit tests for GLYCAM assembly logic (no AmberTools dependency)."""
from __future__ import annotations

import pytest

from poly_csp.forcefield.amber_export import parameterize_selector_fragment
from poly_csp.forcefield.glycam import (
    build_glycam_sequence,
    build_linkage_frcmod,
    build_tleap_script,
)


def test_glycam_sequence_amylose_dp4() -> None:
    seq = build_glycam_sequence("amylose", dp=4)
    assert len(seq) == 4
    assert seq[0] == "4GA"
    assert seq[-1] == "0GA"
    assert all(r == "4GA" for r in seq[:-1])


def test_glycam_sequence_cellulose_dp2() -> None:
    seq = build_glycam_sequence("cellulose", dp=2)
    assert len(seq) == 2
    assert seq[0] == "4GB"
    assert seq[1] == "0GB"


def test_glycam_sequence_dp1() -> None:
    seq = build_glycam_sequence("amylose", dp=1)
    assert seq == ["0GA"]


def test_glycam_sequence_rejects_dp0() -> None:
    with pytest.raises(ValueError, match="dp"):
        build_glycam_sequence("amylose", dp=0)


def test_tleap_script_backbone_only() -> None:
    script = build_tleap_script(polymer="amylose", dp=4)
    assert "GLYCAM_06j" in script
    assert "0GA" in script
    assert "4GA" in script
    assert "saveamberparm" in script
    # Should NOT load GAFF2 leaprc if no selectors
    assert "leaprc.gaff2" not in script
    # Residue names must NOT have nested braces (Bug 2 regression check).
    assert "{ 0GA }" not in script
    assert "sequence { 4GA 4GA 4GA 0GA }" in script


def test_tleap_script_uses_model_name_for_outputs() -> None:
    script = build_tleap_script(polymer="amylose", dp=2, model_name="amylose_ref")
    assert "saveamberparm mol amylose_ref.prmtop amylose_ref.inpcrd" in script


def test_tleap_script_periodic_includes_bond_and_box() -> None:
    script = build_tleap_script(
        polymer="amylose",
        dp=4,
        periodic=True,
        box_vectors_A=(50.0, 50.0, 30.0),
    )
    assert "bond mol.1.C1 mol.4.O4" in script
    assert "setBox" in script
    assert "50.0000" in script


def test_tleap_script_periodic_with_linkage_frcmod() -> None:
    script = build_tleap_script(
        polymer="amylose",
        dp=4,
        periodic=True,
        box_vectors_A=(50.0, 50.0, 30.0),
        linkage_frcmod_path="/tmp/linkage.frcmod",
    )
    assert "loadamberparams /tmp/linkage.frcmod" in script


def test_build_linkage_frcmod(tmp_path) -> None:
    path = build_linkage_frcmod(tmp_path)
    assert path.exists()
    text = path.read_text()
    assert "DIHE" in text
    assert "H2-Cg-Cg-H2" in text
    assert "NONBON" in text


def test_parameterize_selector_fragment_cleans_dummy_atoms(monkeypatch, tmp_path) -> None:
    """Selector fragment inputs should be hydrogen-complete before PDB write."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from poly_csp.forcefield import amber_export as amber_export_mod

    # Build a small mol with a dummy atom
    mol = Chem.MolFromSmiles("[*]C(=O)NC")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)

    written_mols = []

    def fake_write_pdb(m, path):
        written_mols.append(Chem.Mol(m))
        path.write_text("ATOM      1  H   SEL A   1\n", encoding="utf-8")

    def fake_run_command(cmd, cwd, log_path):
        log_path.write_text("OK\n", encoding="utf-8")
        if "antechamber" in cmd:
            (cwd / "selector.mol2").write_text("dummy", encoding="utf-8")
        elif "parmchk2" in cmd:
            (cwd / "selector.frcmod").write_text("dummy", encoding="utf-8")
        elif "tleap" in cmd:
            (cwd / "selector.lib").write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(amber_export_mod, "write_pdb_from_rdkit", fake_write_pdb)
    monkeypatch.setattr(amber_export_mod, "_run_command", fake_run_command)
    monkeypatch.setattr(amber_export_mod, "_ensure_required_tools", lambda tools: None)

    result = parameterize_selector_fragment(mol, work_dir=tmp_path)

    assert len(written_mols) == 1
    written = written_mols[0]
    # No dummy atoms should remain in the molecule written to PDB
    assert all(a.GetAtomicNum() > 0 for a in written.GetAtoms())
    # Hydrogen completion should occur before parameterization.
    assert any(a.GetAtomicNum() == 1 for a in written.GetAtoms())
    amide_n = next(a for a in written.GetAtoms() if a.GetAtomicNum() == 7)
    assert any(nbr.GetAtomicNum() == 1 for nbr in amide_n.GetNeighbors())
    # All paths should be absolute (Bug 3 regression check).
    from pathlib import Path
    for key in ("mol2", "frcmod", "lib"):
        assert Path(result[key]).is_absolute(), f"{key} path is not absolute: {result[key]}"
