from __future__ import annotations

import openmm as mm
from openmm import unit
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.config.schema import SoftSelectorHbondBiasOptions
from poly_csp.forcefield.soft_hbond_bias import (
    add_soft_selector_hbond_bias_force,
    selector_soft_hbond_bias_pairs,
)
from poly_csp.forcefield.system_builder import exclusion_pairs_from_mol
from tests.support import build_forcefield_mol
from poly_csp.topology.selectors import SelectorRegistry


def _toy_bias_mol() -> Chem.Mol:
    rw = Chem.RWMol()
    donor_h = rw.AddAtom(Chem.Atom(1))
    acceptor_o = rw.AddAtom(Chem.Atom(8))
    mol = rw.GetMol()
    donor = mol.GetAtomWithIdx(int(donor_h))
    acceptor = mol.GetAtomWithIdx(int(acceptor_o))
    donor.SetIntProp("_poly_csp_selector_instance", 1)
    donor.SetIntProp("_poly_csp_residue_index", 0)
    donor.SetProp("_poly_csp_connector_role", "amide_n")
    acceptor.SetIntProp("_poly_csp_selector_instance", 2)
    acceptor.SetIntProp("_poly_csp_residue_index", 1)
    acceptor.SetProp("_poly_csp_connector_role", "carbonyl_o")
    return mol


def _energy_for_distance_nm(distance_nm: float) -> float:
    mol = _toy_bias_mol()
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(16.0)
    summary = add_soft_selector_hbond_bias_force(
        system,
        mol,
        options=SoftSelectorHbondBiasOptions(
            enabled=True,
            epsilon_kj_per_mol=3.0,
            r0_nm=0.20,
            half_width_nm=0.05,
            hbond_neighbor_window=1,
        ),
    )
    assert summary["pair_count"] == 1

    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(
        [
            mm.Vec3(0.0, 0.0, 0.0),
            mm.Vec3(float(distance_nm), 0.0, 0.0),
        ]
        * unit.nanometer
    )
    energy = float(
        context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
    )
    del context, integrator
    return energy


def test_selector_soft_hbond_bias_pairs_are_static_across_coordinate_changes() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = build_forcefield_mol(polymer="amylose", dp=2, selector=selector, site="C6")
    excluded = exclusion_pairs_from_mol(mol, exclude_13=True, exclude_14=True)

    baseline_pairs = selector_soft_hbond_bias_pairs(
        mol,
        neighbor_window=1,
        excluded_pairs=excluded,
    )
    assert len(baseline_pairs) == 2

    moved = Chem.Mol(mol)
    conf = moved.GetConformer()
    pos = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, Point3D(float(pos.x + 5.0), float(pos.y), float(pos.z)))
    moved_pairs = selector_soft_hbond_bias_pairs(
        moved,
        neighbor_window=1,
        excluded_pairs=excluded,
    )

    assert moved_pairs == baseline_pairs


def test_add_soft_selector_hbond_bias_force_is_bounded_to_contact_window() -> None:
    short_energy = _energy_for_distance_nm(0.10)
    centered_energy = _energy_for_distance_nm(0.20)
    long_energy = _energy_for_distance_nm(0.30)

    assert abs(short_energy) < 1e-12
    assert centered_energy < -2.9
    assert abs(long_energy) < 1e-12
