from __future__ import annotations

from pathlib import Path

import pytest

from poly_csp.forcefield.export_bundle import prepare_export_bundle
from poly_csp.io.pdbqt import write_receptor_pdbqt
from poly_csp.topology.selectors import SelectorRegistry
from tests.support import build_forcefield_mol, make_fake_runtime_params


def test_write_receptor_pdbqt_uses_runtime_charges_and_supported_atom_types(
    tmp_path: Path,
) -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(dp=1, selector=selector, site="C6")
    runtime_params = make_fake_runtime_params(mol, selector=selector, site="C6")
    bundle = prepare_export_bundle(mol, runtime_params=runtime_params)

    summary = write_receptor_pdbqt(bundle, tmp_path / "receptor.pdbqt")

    assert summary["enabled"] is True
    assert summary["backend"] == "native_openmm_charge_export"
    path = Path(summary["file"])
    assert path.exists()

    atom_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    assert len(atom_lines) == mol.GetNumAtoms()

    parsed_charges = [float(line.split()[-2]) for line in atom_lines]
    parsed_types = {line.split()[-1] for line in atom_lines}
    assert sum(parsed_charges) == pytest.approx(
        sum(item.charge_e for item in bundle.nonbonded_particles),
        abs=5e-3,
    )
    assert {"A", "C", "N", "OA", "HD"}.issubset(parsed_types)
