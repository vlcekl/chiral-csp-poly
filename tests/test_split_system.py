"""Tests for the generic bonded base system used before parameter overlays."""
from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from poly_csp.forcefield.system_builder import (  # noqa: E402
    build_bonded_relaxation_system,
)


def _make_tagged_mol() -> Chem.Mol:
    """Build a small molecule with backbone and 'selector' atoms tagged.

    We use ethanol (CC-O) as backbone and attach a methyl-like group as
    a fake selector for testing.  Total: propanol CH3-CH2-CH2-OH.
    """
    mol = Chem.MolFromSmiles("CCCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)

    # Tag atom 0 (first carbon) as a selector atom.
    atom = mol.GetAtomWithIdx(0)
    atom.SetIntProp("_poly_csp_selector_instance", 0)
    atom.SetIntProp("_poly_csp_selector_local_idx", 0)
    # Also tag its hydrogens.
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            nbr.SetIntProp("_poly_csp_selector_instance", 0)
            nbr.SetIntProp("_poly_csp_selector_local_idx", nbr.GetIdx())

    return mol


def _selector_indices(mol: Chem.Mol) -> set[int]:
    return {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_selector_instance")
    }


def _harmonic_forces(result):
    bond_force = next(
        result.system.getForce(i)
        for i in range(result.system.getNumForces())
        if isinstance(result.system.getForce(i), openmm.HarmonicBondForce)
    )
    angle_force = next(
        result.system.getForce(i)
        for i in range(result.system.getNumForces())
        if isinstance(result.system.getForce(i), openmm.HarmonicAngleForce)
    )
    return bond_force, angle_force


class TestGenericBondedBaseSystem:
    """Verify the base builder already covers selector/backbone junctions."""

    def test_base_system_keeps_all_bonds(self) -> None:
        mol = _make_tagged_mol()
        result = build_bonded_relaxation_system(mol)
        bond_force, _ = _harmonic_forces(result)
        assert bond_force.getNumBonds() == mol.GetNumBonds()

    def test_junction_bonds_are_present_without_special_helper(self) -> None:
        mol = _make_tagged_mol()
        selector_indices = _selector_indices(mol)
        result = build_bonded_relaxation_system(mol)
        bond_force, _ = _harmonic_forces(result)

        for bi in range(bond_force.getNumBonds()):
            p1, p2, _, _ = bond_force.getBondParameters(bi)
            if (p1 in selector_indices) != (p2 in selector_indices):
                return
        raise AssertionError("Junction bond between backbone and selector not found")

    def test_junction_angles_are_present_without_special_helper(self) -> None:
        mol = _make_tagged_mol()
        selector_indices = _selector_indices(mol)
        result = build_bonded_relaxation_system(mol)
        _, angle_force = _harmonic_forces(result)

        for ai in range(angle_force.getNumAngles()):
            p1, p2, p3, _, _ = angle_force.getAngleParameters(ai)
            sides = {
                int(p1) in selector_indices,
                int(p2) in selector_indices,
                int(p3) in selector_indices,
            }
            if len(sides) > 1:
                return
        raise AssertionError("Junction angle between backbone and selector not found")


class TestBackboneFreezing:
    """Verify backbone freezing via mass=0."""

    def test_frozen_backbone_does_not_move(self) -> None:
        """Run dynamics with backbone masses set to 0; verify they don't move."""
        mol = _make_tagged_mol()
        result = build_bonded_relaxation_system(mol)
        system = result.system

        # Identify backbone atoms (no selector tag).
        backbone_indices = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if not atom.HasProp("_poly_csp_selector_instance")
        ]

        # Save initial backbone positions.
        init_xyz = np.asarray(
            result.positions_nm.value_in_unit(unit.nanometer)
        )
        init_backbone = init_xyz[backbone_indices].copy()

        # Freeze backbone.
        for idx in backbone_indices:
            system.setParticleMass(idx, 0.0)

        integrator = openmm.LangevinIntegrator(
            350.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        context = openmm.Context(system, integrator)
        context.setPositions(result.positions_nm)

        integrator.step(200)

        state = context.getState(getPositions=True)
        final_xyz = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        final_backbone = final_xyz[backbone_indices]

        # Backbone atoms should not have moved at all.
        drift = np.max(np.abs(final_backbone - init_backbone))
        assert drift < 1e-6, f"Backbone drifted by {drift} nm despite mass=0"
