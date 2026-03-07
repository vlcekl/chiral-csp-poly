from __future__ import annotations

from pathlib import Path

import openmm as mm
from openmm import app as mmapp, unit

import pytest

from poly_csp.forcefield.amber_export import export_amber_artifacts
from poly_csp.forcefield.export_bundle import prepare_export_bundle
from tests.support import build_forcefield_mol, make_fake_runtime_params


def _charges_from_system(system: mm.System) -> tuple[float, ...]:
    nonbonded = None
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, mm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise AssertionError("Expected NonbondedForce in exported AMBER system.")
    charges = []
    for particle_index in range(nonbonded.getNumParticles()):
        charge, _, _ = nonbonded.getParticleParameters(particle_index)
        charges.append(float(charge.value_in_unit(unit.elementary_charge)))
    return tuple(charges)


def test_export_amber_artifacts_roundtrips_runtime_system(tmp_path: Path) -> None:
    mol = build_forcefield_mol(polymer="amylose", dp=2)
    runtime_params = make_fake_runtime_params(mol)
    bundle = prepare_export_bundle(mol, runtime_params=runtime_params)

    summary = export_amber_artifacts(bundle, tmp_path, model_name="model")

    assert summary["enabled"] is True
    assert summary["parameter_backend"] == "runtime_system_export"
    prmtop_path = Path(summary["files"]["prmtop"])
    inpcrd_path = Path(summary["files"]["inpcrd"])
    assert prmtop_path.exists()
    assert inpcrd_path.exists()
    assert Path(summary["manifest"]).exists()

    prmtop = mmapp.AmberPrmtopFile(str(prmtop_path))
    amber_system = prmtop.createSystem(nonbondedMethod=mmapp.NoCutoff)
    exported_charges = _charges_from_system(amber_system)

    assert len(exported_charges) == mol.GetNumAtoms()
    assert sum(exported_charges) == pytest.approx(
        sum(item.charge_e for item in bundle.nonbonded_particles),
        abs=1e-6,
    )
