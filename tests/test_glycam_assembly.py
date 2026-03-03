# tests/test_glycam_assembly.py
"""Unit tests for GLYCAM assembly logic (no AmberTools dependency)."""
from __future__ import annotations

import pytest

from poly_csp.io.glycam_assembly import (
    build_glycam_sequence,
    build_tleap_script,
)


def test_glycam_sequence_amylose_dp4() -> None:
    seq = build_glycam_sequence("amylose", dp=4)
    assert len(seq) == 4
    assert seq[0] == "0GA"
    assert seq[-1] == "4GA"
    assert all(r == "4GA" for r in seq[1:])


def test_glycam_sequence_cellulose_dp2() -> None:
    seq = build_glycam_sequence("cellulose", dp=2)
    assert len(seq) == 2
    assert seq[0] == "0GB"
    assert seq[1] == "4GB"


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


def test_tleap_script_with_selector() -> None:
    script = build_tleap_script(
        polymer="amylose",
        dp=4,
        selector_lib_path="/tmp/sel.lib",
        selector_frcmod_path="/tmp/sel.frcmod",
    )
    assert "GLYCAM_06j" in script
    assert "gaff2" in script
    assert "/tmp/sel.lib" in script
    assert "/tmp/sel.frcmod" in script
    # Should have both GLYCAM and GAFF2
    assert "0GA" in script


def test_parameterize_selector_fragment_cleans_dummy_atoms(monkeypatch, tmp_path) -> None:
    """Dummy atoms in the selector mol should be replaced with H before PDB write."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from poly_csp.io import glycam_assembly

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

    monkeypatch.setattr(glycam_assembly, "write_pdb_from_rdkit", fake_write_pdb)
    monkeypatch.setattr(glycam_assembly, "_run_command", fake_run_command)
    monkeypatch.setattr(glycam_assembly, "_ensure_required_tools", lambda tools: None)

    glycam_assembly.parameterize_selector_fragment(mol, work_dir=tmp_path)

    assert len(written_mols) == 1
    written = written_mols[0]
    # No dummy atoms should remain in the molecule written to PDB
    assert all(a.GetAtomicNum() > 0 for a in written.GetAtoms())

