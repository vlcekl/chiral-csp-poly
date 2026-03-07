from __future__ import annotations

from types import SimpleNamespace

import openmm as mm
import numpy as np
import pytest
from openmm import unit

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.relaxation import RelaxSpec, run_staged_relaxation
from poly_csp.forcefield.system_builder import SystemBuildResult
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.structure.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.terminals import apply_terminal_mode


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-4.71238898038469,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=4.933333333333334,
        handedness="left",
    )


def _forcefield_selector_mol():
    selector = make_35_dmpc_template()
    template = make_glucose_template("amylose", monomer_representation="anhydro")
    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    topology = apply_terminal_mode(
        mol=topology,
        mode="open",
        caps={},
        representation="anhydro",
    )
    structure = build_backbone_structure(topology, _helix()).mol
    structure = attach_selector(
        mol_polymer=structure,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    return build_forcefield_molecule(structure).mol, selector


def _fake_system_result(mol, nonbonded_mode: str) -> SystemBuildResult:
    system = mm.System()
    for atom in mol.GetAtoms():
        mass = 1.0 if atom.GetAtomicNum() == 1 else 12.0
        system.addParticle(mass)

    bond_force = mm.HarmonicBondForce()
    for bond in mol.GetBonds():
        bond_force.addBond(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            0.15,
            50.0,
        )
    system.addForce(bond_force)

    if nonbonded_mode == "soft":
        repulsive = mm.CustomNonbondedForce("0")
        repulsive.addPerParticleParameter("sigma")
        for _ in mol.GetAtoms():
            repulsive.addParticle([0.2])
        system.addForce(repulsive)
        exception_summary = {"mode": "soft", "num_exclusions": 0}
    else:
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        for _ in mol.GetAtoms():
            nonbonded.addParticle(0.0, 0.2, 0.0)
        system.addForce(nonbonded)
        exception_summary = {"exceptions_seen": 0, "exceptions_patched": 0}

    xyz = np.asarray(build_forcefield_molecule(mol).mol.GetConformer(0).GetPositions(), dtype=float)
    positions_nm = (xyz / 10.0) * unit.nanometer
    return SystemBuildResult(
        system=system,
        positions_nm=positions_nm,
        excluded_pairs=set(),
        nonbonded_mode=nonbonded_mode,
        topology_manifest=(),
        component_counts={"backbone": 1, "selector": 1, "connector": 1},
        exception_summary=exception_summary,
        source_manifest={"fake": True},
    )


def test_run_staged_relaxation_uses_soft_then_full_runtime_systems(monkeypatch) -> None:
    mol, selector = _forcefield_selector_mol()
    runtime = SimpleNamespace(
        glycam=None,
        selector_params_by_name={},
        connector_params_by_key={},
        source_manifest={"runtime": {"cache": {"kind": "test"}}},
    )
    calls: list[str] = []

    def fake_create_system(*args, nonbonded_mode, **kwargs):
        calls.append(nonbonded_mode)
        return _fake_system_result(mol, nonbonded_mode=nonbonded_mode)

    monkeypatch.setattr("poly_csp.forcefield.relaxation.create_system", fake_create_system)

    spec = RelaxSpec(
        enabled=True,
        positional_k=10.0,
        dihedral_k=0.0,
        hbond_k=0.0,
        n_stages=1,
        max_iterations=5,
        freeze_backbone=False,
        anneal_enabled=False,
    )
    relaxed, summary = run_staged_relaxation(
        mol=mol,
        spec=spec,
        selector=selector,
        runtime_params=runtime,
    )

    assert calls == ["soft", "full"]
    assert relaxed.GetNumAtoms() == mol.GetNumAtoms()
    assert relaxed.GetNumConformers() == 1
    assert summary["enabled"] is True
    assert summary["protocol"] == "two_stage_runtime"
    assert summary["stage1_nonbonded_mode"] == "soft"
    assert summary["stage2_nonbonded_mode"] == "full"
    assert summary["source_manifest"] == {"fake": True}
