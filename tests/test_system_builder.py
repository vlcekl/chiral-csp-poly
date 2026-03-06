from __future__ import annotations

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.connectors import (
    build_capped_monomer_fragment,
    extract_linkage_params_from_system,
)
from poly_csp.forcefield.system_builder import create_system
from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.topology.atom_mapping import build_atom_map
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def _selector_core_terms(fragment) -> tuple[tuple[int, int], tuple[int, int, int], tuple[int, int, int, int]]:
    selector_core = {
        idx
        for role, idx in fragment.atom_roles.items()
        if role.startswith("SL_") and role not in set(fragment.connector_atom_roles.values())
    }
    idx_to_role = {idx: role for role, idx in fragment.atom_roles.items()}

    bond = None
    for rd_bond in fragment.mol.GetBonds():
        a = rd_bond.GetBeginAtomIdx()
        b = rd_bond.GetEndAtomIdx()
        if a in selector_core and b in selector_core:
            bond = (a, b)
            break
    if bond is None:
        raise AssertionError("Expected a selector-core bond in capped fragment.")

    angle = None
    for center in selector_core:
        nbrs = [
            nbr.GetIdx()
            for nbr in fragment.mol.GetAtomWithIdx(center).GetNeighbors()
            if nbr.GetIdx() in selector_core
        ]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                angle = (nbrs[i], center, nbrs[j])
                break
            if angle is not None:
                break
        if angle is not None:
            break
    if angle is None:
        raise AssertionError("Expected a selector-core angle in capped fragment.")

    torsion = None
    for rd_bond in fragment.mol.GetBonds():
        b = rd_bond.GetBeginAtomIdx()
        c = rd_bond.GetEndAtomIdx()
        if b not in selector_core or c not in selector_core:
            continue
        left = [
            nbr.GetIdx()
            for nbr in fragment.mol.GetAtomWithIdx(b).GetNeighbors()
            if nbr.GetIdx() != c and nbr.GetIdx() in selector_core
        ]
        right = [
            nbr.GetIdx()
            for nbr in fragment.mol.GetAtomWithIdx(c).GetNeighbors()
            if nbr.GetIdx() != b and nbr.GetIdx() in selector_core
        ]
        for a in left:
            for d in right:
                if len({a, b, c, d}) == 4:
                    torsion = (a, b, c, d)
                    break
            if torsion is not None:
                break
        if torsion is not None:
            break
    if torsion is None:
        roles = sorted(idx_to_role[idx] for idx in selector_core)
        raise AssertionError(f"Expected a selector-core torsion in capped fragment: {roles}")

    return bond, angle, torsion


def test_create_system_particle_count_matches_molecule() -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, build_backbone_coords(template, _helix(), dp=2))
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    atom_map = build_atom_map(mol)
    system = create_system(mol, atom_map=atom_map)

    assert system.getNumParticles() == mol.GetNumAtoms()
    assert system.getNumForces() > 0


def test_create_system_applies_selector_gaff_parameters_without_duplicate_bonds() -> None:
    import openmm as mm
    from openmm import unit

    fragment = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_35_dmpc_template(),
        site="C6",
    )
    (bond_a, bond_b), (angle_a, angle_b, angle_c), torsion = _selector_core_terms(fragment)

    gaff_bond = mm.HarmonicBondForce()
    gaff_bond.addBond(
        bond_a,
        bond_b,
        0.142 * unit.nanometer,
        432.0 * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    gaff_angle = mm.HarmonicAngleForce()
    gaff_angle.addAngle(
        angle_a,
        angle_b,
        angle_c,
        2.21 * unit.radian,
        87.0 * unit.kilojoule_per_mole / unit.radian**2,
    )
    gaff_torsion = mm.PeriodicTorsionForce()
    gaff_torsion.addTorsion(
        torsion[0],
        torsion[1],
        torsion[2],
        torsion[3],
        3,
        0.25 * unit.radian,
        7.5 * unit.kilojoule_per_mole,
    )

    system = create_system(
        fragment.mol,
        gaff_params={"forces": [gaff_bond, gaff_angle, gaff_torsion]},
    )

    harmonic_bond = next(
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), mm.HarmonicBondForce)
    )
    harmonic_angle = next(
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), mm.HarmonicAngleForce)
    )
    torsions = [
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), mm.PeriodicTorsionForce)
    ]

    matching_bonds = 0
    for idx in range(harmonic_bond.getNumBonds()):
        a, b, r0, k = harmonic_bond.getBondParameters(idx)
        if {int(a), int(b)} == {bond_a, bond_b}:
            matching_bonds += 1
            assert abs(float(r0.value_in_unit(unit.nanometer)) - 0.142) < 1e-8
            assert (
                abs(
                    float(
                        k.value_in_unit(
                            unit.kilojoule_per_mole / unit.nanometer**2
                        )
                    )
                    - 432.0
                )
                < 1e-8
            )
    assert matching_bonds == 1

    matching_angles = 0
    for idx in range(harmonic_angle.getNumAngles()):
        a, b, c, theta0, k = harmonic_angle.getAngleParameters(idx)
        if (int(a), int(b), int(c)) == (angle_a, angle_b, angle_c) or (
            int(c),
            int(b),
            int(a),
        ) == (angle_a, angle_b, angle_c):
            matching_angles += 1
            assert abs(float(theta0.value_in_unit(unit.radian)) - 2.21) < 1e-8
            assert (
                abs(
                    float(
                        k.value_in_unit(
                            unit.kilojoule_per_mole / unit.radian**2
                        )
                    )
                    - 87.0
                )
                < 1e-8
            )
    assert matching_angles == 1

    assert len(torsions) == 1
    assert torsions[0].getNumTorsions() == 1


def test_create_system_applies_connector_parameters_without_duplicate_bonds() -> None:
    import openmm as mm
    from openmm import unit

    fragment = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_35_dmpc_template(),
        site="C6",
    )
    ref_system = mm.System()
    for _ in range(fragment.mol.GetNumAtoms()):
        ref_system.addParticle(12.0)

    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    torsion_force = mm.PeriodicTorsionForce()

    carbonyl_role = fragment.connector_atom_roles["carbonyl_c"]
    carbonyl_idx = fragment.atom_roles[carbonyl_role]
    amide_idx = fragment.atom_roles[fragment.connector_atom_roles["amide_n"]]

    bond_force.addBond(fragment.atom_roles["BB_O6"], carbonyl_idx, 0.136, 777.0)
    angle_force.addAngle(fragment.atom_roles["BB_C6"], fragment.atom_roles["BB_O6"], carbonyl_idx, 2.02, 66.0)
    torsion_force.addTorsion(fragment.atom_roles["BB_C6"], fragment.atom_roles["BB_O6"], carbonyl_idx, amide_idx, 2, 3.14, 9.0)
    ref_system.addForce(bond_force)
    ref_system.addForce(angle_force)
    ref_system.addForce(torsion_force)

    params = extract_linkage_params_from_system(ref_system=ref_system, fragment=fragment)
    system = create_system(fragment.mol, connector_params=params)

    harmonic_bond = next(
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), mm.HarmonicBondForce)
    )
    harmonic_angle = next(
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), mm.HarmonicAngleForce)
    )
    torsions = [
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), mm.PeriodicTorsionForce)
    ]

    matching_bonds = 0
    for bi in range(harmonic_bond.getNumBonds()):
        a, b, r0, k = harmonic_bond.getBondParameters(bi)
        if {int(a), int(b)} == {fragment.atom_roles["BB_O6"], carbonyl_idx}:
            matching_bonds += 1
            assert abs(float(r0.value_in_unit(unit.nanometer)) - 0.136) < 1e-8
            assert (
                abs(
                    float(
                        k.value_in_unit(
                            unit.kilojoule_per_mole / unit.nanometer**2
                        )
                    )
                    - 777.0
                )
                < 1e-8
            )
    assert matching_bonds == 1

    matching_angles = 0
    for ai in range(harmonic_angle.getNumAngles()):
        a, b, c, theta0, k = harmonic_angle.getAngleParameters(ai)
        if (int(a), int(b), int(c)) == (
            fragment.atom_roles["BB_C6"],
            fragment.atom_roles["BB_O6"],
            carbonyl_idx,
        ) or (int(c), int(b), int(a)) == (
            fragment.atom_roles["BB_C6"],
            fragment.atom_roles["BB_O6"],
            carbonyl_idx,
        ):
            matching_angles += 1
            assert abs(float(theta0.value_in_unit(unit.radian)) - 2.02) < 1e-8
            assert (
                abs(
                    float(
                        k.value_in_unit(
                            unit.kilojoule_per_mole / unit.radian**2
                        )
                    )
                    - 66.0
                )
                < 1e-8
            )
    assert matching_angles == 1

    assert len(torsions) == 1
    assert torsions[0].getNumTorsions() == 1
