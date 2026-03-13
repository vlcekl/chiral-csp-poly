from __future__ import annotations

from collections import defaultdict

import openmm as mm
from openmm import unit
from rdkit import Chem

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.connectors import (
    ConnectorAngleTemplate,
    ConnectorAtomParams,
    ConnectorBondTemplate,
    ConnectorParams,
    ConnectorToken,
    ConnectorTorsionTemplate,
)
from poly_csp.forcefield.gaff import SelectorAtomParams, SelectorFragmentParams
from poly_csp.forcefield.glycam import (
    GlycamAtomParams,
    GlycamAtomToken,
    GlycamBondTemplate,
    GlycamLinkageTemplate,
    GlycamParams,
    GlycamResidueTemplate,
    glycam_residue_roles_for_dp,
)
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.system_builder import create_system
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.structure.pbc import ensure_periodic_box_vectors
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.terminals import apply_terminal_mode


_GLYCAM_HYDROGEN_ALIASES = {
    "HO1": "H1O",
    "HO2": "H2O",
    "HO3": "H3O",
    "HO4": "H4O",
    "HO6": "H6O",
}


def _helix() -> HelixSpec:
    return HelixSpec(
        name="system_builder_test_helix",
        theta_rad=-4.71238898038469,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=4.933333333333334,
        handedness="left",
    )


def _build_forcefield_mol(
    *,
    polymer: str = "amylose",
    dp: int = 2,
    selector=None,
    site: str = "C6",
    end_mode: str = "open",
    end_caps: dict[str, str] | None = None,
) -> Chem.Mol:
    template = make_glucose_template(polymer, monomer_representation="anhydro")
    topology = polymerize(
        template=template,
        dp=dp,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    topology = apply_terminal_mode(
        mol=topology,
        mode=end_mode,  # type: ignore[arg-type]
        caps=dict(end_caps or {}),
        representation="anhydro",
    )
    structure = build_backbone_structure(topology, _helix()).mol
    if selector is not None:
        for residue_index in range(dp):
            structure = attach_selector(
                mol_polymer=structure,
                residue_index=residue_index,
                site=site,  # type: ignore[arg-type]
                selector=selector,
            )
    if end_mode == "periodic":
        ensure_periodic_box_vectors(
            structure,
            _helix(),
            dp=dp,
            padding_A=30.0,
        )
    return build_forcefield_molecule(structure).mol


def _glycam_name(atom_name: str) -> str:
    return _GLYCAM_HYDROGEN_ALIASES.get(atom_name, atom_name)


def _snapshot_bonds(force: mm.HarmonicBondForce) -> tuple[tuple[int, int, float, float], ...]:
    out = []
    for i in range(force.getNumBonds()):
        a, b, r0, k = force.getBondParameters(i)
        left, right = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
        out.append(
            (
                left,
                right,
                float(r0.value_in_unit(unit.nanometer)),
                float(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)),
            )
        )
    return tuple(sorted(out))


def _snapshot_angles(force: mm.HarmonicAngleForce) -> tuple[tuple[int, int, int, float, float], ...]:
    out = []
    for i in range(force.getNumAngles()):
        a, b, c, theta0, k = force.getAngleParameters(i)
        if int(a) <= int(c):
            key = (int(a), int(b), int(c))
        else:
            key = (int(c), int(b), int(a))
        out.append(
            (
                *key,
                float(theta0.value_in_unit(unit.radian)),
                float(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)),
            )
        )
    return tuple(sorted(out))


def _snapshot_torsions(
    force: mm.PeriodicTorsionForce,
) -> tuple[tuple[int, int, int, int, int, float, float], ...]:
    out = []
    for i in range(force.getNumTorsions()):
        a, b, c, d, periodicity, phase, k = force.getTorsionParameters(i)
        forward = (int(a), int(b), int(c), int(d))
        reverse = (int(d), int(c), int(b), int(a))
        key = forward if forward <= reverse else reverse
        out.append(
            (
                *key,
                int(periodicity),
                float(phase.value_in_unit(unit.radian)),
                float(k.value_in_unit(unit.kilojoule_per_mole)),
            )
        )
    return tuple(sorted(out))


def _bonded_snapshot(result) -> dict[str, tuple]:
    bonds = ()
    angles = ()
    torsions = ()
    for i in range(result.system.getNumForces()):
        force = result.system.getForce(i)
        if isinstance(force, mm.HarmonicBondForce):
            bonds = _snapshot_bonds(force)
        elif isinstance(force, mm.HarmonicAngleForce):
            angles = _snapshot_angles(force)
        elif isinstance(force, mm.PeriodicTorsionForce):
            torsions = _snapshot_torsions(force)
    return {
        "bonds": bonds,
        "angles": angles,
        "torsions": torsions,
    }


def _soft_nonbonded_force(result) -> mm.CustomNonbondedForce:
    return next(
        force
        for force in (result.system.getForce(i) for i in range(result.system.getNumForces()))
        if isinstance(force, mm.CustomNonbondedForce)
    )


def _fake_glycam_params(mol: Chem.Mol) -> GlycamParams:
    polymer = mol.GetProp("_poly_csp_polymer")
    representation = mol.GetProp("_poly_csp_representation")
    end_mode = mol.GetProp("_poly_csp_end_mode")
    dp = int(mol.GetIntProp("_poly_csp_dp"))
    role_end_mode = end_mode if end_mode in {"open", "periodic"} else "open"
    residue_roles = glycam_residue_roles_for_dp(dp, end_mode=role_end_mode)  # type: ignore[arg-type]
    residue_names = {
        "amylose": {
            "terminal_reducing": "4GA",
            "internal": "4GA",
            "terminal_nonreducing": "0GA",
            "periodic": "4GA",
        },
        "cellulose": {
            "terminal_reducing": "4GB",
            "internal": "4GB",
            "terminal_nonreducing": "0GB",
            "periodic": "4GB",
        },
    }[polymer]

    atom_indices_by_residue: dict[int, dict[str, int]] = defaultdict(dict)
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_manifest_source"):
            continue
        if atom.GetProp("_poly_csp_manifest_source") != "backbone":
            continue
        residue_index = int(atom.GetIntProp("_poly_csp_residue_index"))
        atom_name = atom.GetProp("_poly_csp_atom_name")
        glycam_name = _glycam_name(atom_name)
        atom_indices_by_residue[residue_index][glycam_name] = int(atom.GetIdx())

    atom_params: dict[tuple[str, str], GlycamAtomParams] = {}
    residue_templates: dict[str, GlycamResidueTemplate] = {}
    residue_bonds: dict[str, dict[tuple[tuple[int, str], tuple[int, str]], GlycamBondTemplate]] = defaultdict(dict)

    for residue_index, residue_role in enumerate(residue_roles):
        atom_names = tuple(sorted(atom_indices_by_residue[residue_index]))
        residue_templates[residue_role] = GlycamResidueTemplate(
            residue_role=residue_role,
            residue_name=residue_names[residue_role],
            atom_names=atom_names,
            bonds=(),
            angles=(),
            torsions=(),
        )
        for atom_name in atom_names:
            atom_params[(residue_role, atom_name)] = GlycamAtomParams(
                charge_e=0.0,
                sigma_nm=0.3,
                epsilon_kj_per_mol=0.1,
                residue_name=residue_names[residue_role],
                source_atom_name=atom_name,
            )

    linkage_bonds: dict[
        tuple[str, str],
        dict[tuple[tuple[int, str], tuple[int, str]], GlycamBondTemplate],
    ] = defaultdict(dict)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if (
            begin.HasProp("_poly_csp_manifest_source")
            and end.HasProp("_poly_csp_manifest_source")
            and begin.GetProp("_poly_csp_manifest_source") == "backbone"
            and end.GetProp("_poly_csp_manifest_source") == "backbone"
        ):
            res_i = int(begin.GetIntProp("_poly_csp_residue_index"))
            res_j = int(end.GetIntProp("_poly_csp_residue_index"))
            name_i = _glycam_name(begin.GetProp("_poly_csp_atom_name"))
            name_j = _glycam_name(end.GetProp("_poly_csp_atom_name"))
            token_i = GlycamAtomToken(residue_offset=0, atom_name=name_i)
            if res_i == res_j:
                token_j = GlycamAtomToken(residue_offset=0, atom_name=name_j)
                key = ((token_i.residue_offset, token_i.atom_name), (token_j.residue_offset, token_j.atom_name))
                if key[0] > key[1]:
                    key = (key[1], key[0])
                    token_i, token_j = token_j, token_i
                residue_bonds[residue_roles[res_i]][key] = GlycamBondTemplate(
                    atoms=(token_i, token_j),
                    length_nm=0.15,
                    k_kj_per_mol_nm2=100.0,
                )
                continue

            if end_mode == "periodic" and {res_i, res_j} == {0, dp - 1}:
                left_residue = dp - 1
                right_residue = 0
            else:
                left_residue = min(res_i, res_j)
                right_residue = max(res_i, res_j)
            left_token = GlycamAtomToken(
                residue_offset=0,
                atom_name=name_i if res_i == left_residue else name_j,
            )
            right_token = GlycamAtomToken(
                residue_offset=1,
                atom_name=name_j if res_j == right_residue else name_i,
            )
            key = (
                (left_token.residue_offset, left_token.atom_name),
                (right_token.residue_offset, right_token.atom_name),
            )
            linkage_bonds[(residue_roles[left_residue], residue_roles[right_residue])][key] = GlycamBondTemplate(
                atoms=(left_token, right_token),
                length_nm=0.14,
                k_kj_per_mol_nm2=120.0,
            )

    for residue_role, template in list(residue_templates.items()):
        residue_templates[residue_role] = GlycamResidueTemplate(
            residue_role=template.residue_role,
            residue_name=template.residue_name,
            atom_names=template.atom_names,
            bonds=tuple(sorted(residue_bonds[residue_role].values(), key=lambda item: item.atoms)),
            angles=(),
            torsions=(),
        )

    linkage_templates = {
        pair: GlycamLinkageTemplate(
            residue_roles=pair,  # type: ignore[arg-type]
            bonds=tuple(sorted(bonds.values(), key=lambda item: item.atoms)),
            angles=(),
            torsions=(),
        )
        for pair, bonds in linkage_bonds.items()
    }

    return GlycamParams(
        polymer=polymer,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        end_mode=end_mode,  # type: ignore[arg-type]
        atom_params=atom_params,
        residue_templates=residue_templates,
        linkage_templates=linkage_templates,
        supported_states=tuple(
            (polymer, representation, end_mode, role)
            for role in residue_templates
        ),
        provenance={"parameter_backend": "test_fake_glycam"},
    )


def _selector_names(mol: Chem.Mol) -> list[str]:
    return [
        atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "selector"
    ]


def _connector_role_map(mol: Chem.Mol) -> dict[str, str]:
    return {
        atom.GetProp("_poly_csp_connector_role"): atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "connector"
        and atom.HasProp("_poly_csp_connector_role")
    }


def _fake_selector_params(mol: Chem.Mol, selector_name: str) -> SelectorFragmentParams:
    atom_names = _selector_names(mol)
    bonds = ()
    angles = ()
    torsions = ()
    if len(atom_names) >= 2:
        bonds = (
            _selector_bond_template(atom_names[0], atom_names[1]),
        )
    if len(atom_names) >= 3:
        angles = (
            _selector_angle_template(atom_names[0], atom_names[1], atom_names[2]),
        )
    if len(atom_names) >= 4:
        torsions = (
            _selector_torsion_template(
                atom_names[0], atom_names[1], atom_names[2], atom_names[3]
            ),
        )
    return SelectorFragmentParams(
        selector_name=selector_name,
        atom_params={
            atom_name: SelectorAtomParams(
                atom_name=atom_name,
                charge_e=-0.05,
                sigma_nm=0.31,
                epsilon_kj_per_mol=0.12,
            )
            for atom_name in atom_names
        },
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        source_prmtop="selector_fragment.prmtop",
        fragment_atom_count=len(atom_names),
    )


def _selector_bond_template(a: str, b: str):
    from poly_csp.forcefield.gaff import SelectorBondTemplate

    return SelectorBondTemplate(atom_names=(a, b), length_nm=0.145, k_kj_per_mol_nm2=220.0)


def _selector_angle_template(a: str, b: str, c: str):
    from poly_csp.forcefield.gaff import SelectorAngleTemplate

    return SelectorAngleTemplate(
        atom_names=(a, b, c),
        theta0_rad=2.09,
        k_kj_per_mol_rad2=60.0,
    )


def _selector_torsion_template(a: str, b: str, c: str, d: str):
    from poly_csp.forcefield.gaff import SelectorTorsionTemplate

    return SelectorTorsionTemplate(
        atom_names=(a, b, c, d),
        periodicity=2,
        phase_rad=3.14,
        k_kj_per_mol=4.0,
    )


def _fake_connector_params(mol: Chem.Mol, *, selector_name: str, linkage_type: str, site: str) -> ConnectorParams:
    connector_names = [
        atom.GetProp("_poly_csp_atom_name")
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_manifest_source")
        and atom.GetProp("_poly_csp_manifest_source") == "connector"
    ]
    selector_names = _selector_names(mol)
    role_map = _connector_role_map(mol)
    anchor = f"O{site[1:]}"
    carbon = site
    torsions: tuple[ConnectorTorsionTemplate, ...] = ()
    if linkage_type == "carbamate":
        torsions = (
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("backbone", carbon),
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("connector", role_map["amide_n"]),
                ),
                periodicity=2,
                phase_rad=3.14,
                k_kj_per_mol=8.0,
            ),
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("connector", role_map["amide_n"]),
                    ConnectorToken("selector", selector_names[0]),
                ),
                periodicity=2,
                phase_rad=0.0,
                k_kj_per_mol=7.0,
            ),
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("connector", role_map["carbonyl_o"]),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("connector", role_map["amide_n"]),
                    ConnectorToken("selector", selector_names[1]),
                ),
                periodicity=2,
                phase_rad=0.0,
                k_kj_per_mol=6.0,
            ),
        )
    else:
        torsions = (
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("connector", role_map["carbonyl_o"]),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("selector", selector_names[0]),
                    ConnectorToken("selector", selector_names[1]),
                ),
                periodicity=2,
                phase_rad=0.0,
                k_kj_per_mol=6.0,
            ),
            ConnectorTorsionTemplate(
                atoms=(
                    ConnectorToken("backbone", carbon),
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                    ConnectorToken("selector", selector_names[0]),
                ),
                periodicity=2,
                phase_rad=3.14,
                k_kj_per_mol=5.0,
            ),
        )
    return ConnectorParams(
        polymer=mol.GetProp("_poly_csp_polymer"),  # type: ignore[arg-type]
        selector_name=selector_name,
        site=site,  # type: ignore[arg-type]
        monomer_representation="natural_oh",
        linkage_type=linkage_type,  # type: ignore[arg-type]
        atom_params={
            atom_name: ConnectorAtomParams(
                atom_name=atom_name,
                charge_e=0.04,
                sigma_nm=0.29,
                epsilon_kj_per_mol=0.09,
            )
            for atom_name in connector_names
        },
        connector_role_atom_names=role_map,
        bonds=(
            ConnectorBondTemplate(
                atoms=(
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                ),
                length_nm=0.136,
                k_kj_per_mol_nm2=640.0,
            ),
        ),
        angles=(
            ConnectorAngleTemplate(
                atoms=(
                    ConnectorToken("backbone", carbon),
                    ConnectorToken("backbone", anchor),
                    ConnectorToken("connector", role_map["carbonyl_c"]),
                ),
                theta0_rad=2.04,
                k_kj_per_mol_rad2=77.0,
            ),
        ),
        torsions=torsions,
        source_prmtop="connector_fragment.prmtop",
        fragment_atom_count=len(connector_names),
    )


def test_create_system_soft_and_full_share_bonded_terms_for_pure_backbone() -> None:
    mol = _build_forcefield_mol(polymer="amylose", dp=2)
    glycam = _fake_glycam_params(mol)

    soft = create_system(
        mol,
        glycam_params=glycam,
        nonbonded_mode="soft",
    )
    full = create_system(
        mol,
        glycam_params=glycam,
        nonbonded_mode="full",
    )

    assert soft.bonded_term_summary == full.bonded_term_summary
    assert soft.excluded_pairs == full.excluded_pairs
    assert _bonded_snapshot(soft) == _bonded_snapshot(full)
    assert soft.force_inventory.counts.get("CustomNonbondedForce", 0) == 1
    assert soft.force_inventory.counts.get("NonbondedForce", 0) == 0
    assert full.force_inventory.counts.get("NonbondedForce", 0) == 1
    assert full.force_inventory.counts.get("CustomNonbondedForce", 0) == 0
    assert full.exception_summary["expected_14_pairs"] > 0
    assert full.exception_summary["expected_14_pairs"] == full.exception_summary["found_14_pairs"]
    assert full.exception_summary["counts_by_rule_bucket"]["backbone_backbone"] > 0
    assert full.exception_summary["patched_14_pairs"] == 0


def test_create_system_soft_and_full_share_bonded_terms_for_mixed_system() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = _build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    glycam = _fake_glycam_params(mol)
    selector_params = _fake_selector_params(mol, selector.name)
    connector_params = _fake_connector_params(
        mol,
        selector_name=selector.name,
        linkage_type=selector.linkage_type,
        site="C6",
    )

    soft = create_system(
        mol,
        glycam_params=glycam,
        selector_params_by_name={selector.name: selector_params},
        connector_params_by_key={(selector.name, "C6"): connector_params},
        nonbonded_mode="soft",
    )
    full = create_system(
        mol,
        glycam_params=glycam,
        selector_params_by_name={selector.name: selector_params},
        connector_params_by_key={(selector.name, "C6"): connector_params},
        nonbonded_mode="full",
    )

    assert soft.bonded_term_summary == full.bonded_term_summary
    assert soft.excluded_pairs == full.excluded_pairs
    assert _bonded_snapshot(soft) == _bonded_snapshot(full)
    assert soft.topology_manifest == full.topology_manifest
    assert soft.component_counts == full.component_counts
    assert soft.bonded_term_summary.by_source["selector"]["bonds"] > 0
    assert soft.bonded_term_summary.by_source["connector"]["torsions"] > 0
    assert soft.force_inventory.counts.get("CustomNonbondedForce", 0) == 1
    assert full.force_inventory.counts.get("NonbondedForce", 0) == 1
    assert full.exception_summary["expected_14_pairs"] == full.exception_summary["found_14_pairs"]
    assert full.exception_summary["counts_by_rule_bucket"]["selector_selector"] > 0
    assert full.exception_summary["connector_involving_pairs"] > 0
    assert full.exception_summary["patched_14_pairs"] > 0


def test_create_system_supports_ester_connectors_in_both_modes() -> None:
    selector = SelectorRegistry.get("tmb")
    mol = _build_forcefield_mol(polymer="cellulose", dp=1, selector=selector, site="C3")
    glycam = _fake_glycam_params(mol)
    selector_params = _fake_selector_params(mol, selector.name)
    connector_params = _fake_connector_params(
        mol,
        selector_name=selector.name,
        linkage_type=selector.linkage_type,
        site="C3",
    )

    soft = create_system(
        mol,
        glycam_params=glycam,
        selector_params_by_name={selector.name: selector_params},
        connector_params_by_key={(selector.name, "C3"): connector_params},
        nonbonded_mode="soft",
    )
    full = create_system(
        mol,
        glycam_params=glycam,
        selector_params_by_name={selector.name: selector_params},
        connector_params_by_key={(selector.name, "C3"): connector_params},
        nonbonded_mode="full",
    )

    assert soft.bonded_term_summary == full.bonded_term_summary
    assert soft.bonded_term_summary.by_source["connector"]["torsions"] > 0
    assert soft.exception_summary["force_kind"] == "CustomNonbondedForce"
    assert full.exception_summary["force_kind"] == "NonbondedForce"
    assert full.exception_summary["counts_by_rule_bucket"]["cross_boundary"] > 0


def test_create_system_soft_mode_can_exclude_14_pairs() -> None:
    mol = _build_forcefield_mol(polymer="amylose", dp=2)
    glycam = _fake_glycam_params(mol)

    soft_default = create_system(
        mol,
        glycam_params=glycam,
        nonbonded_mode="soft",
    )
    soft_ex14 = create_system(
        mol,
        glycam_params=glycam,
        nonbonded_mode="soft",
        soft_exclude_14=True,
    )
    full = create_system(
        mol,
        glycam_params=glycam,
        nonbonded_mode="full",
        soft_exclude_14=True,
    )

    assert len(soft_ex14.excluded_pairs) > len(soft_default.excluded_pairs)
    assert soft_default.exception_summary["soft_exclude_14"] is False
    assert soft_ex14.exception_summary["soft_exclude_14"] is True
    assert full.excluded_pairs == soft_default.excluded_pairs


def test_create_system_soft_mode_scales_selector_aromatic_sigmas_only() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = _build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    glycam = _fake_glycam_params(mol)
    selector_params = _fake_selector_params(mol, selector.name)
    connector_params = _fake_connector_params(
        mol,
        selector_name=selector.name,
        linkage_type=selector.linkage_type,
        site="C6",
    )

    baseline = create_system(
        mol,
        glycam_params=glycam,
        selector_params_by_name={selector.name: selector_params},
        connector_params_by_key={(selector.name, "C6"): connector_params},
        nonbonded_mode="soft",
    )
    scaled = create_system(
        mol,
        glycam_params=glycam,
        selector_params_by_name={selector.name: selector_params},
        connector_params_by_key={(selector.name, "C6"): connector_params},
        nonbonded_mode="soft",
        anti_stacking_sigma_scale=1.5,
    )

    baseline_force = _soft_nonbonded_force(baseline)
    scaled_force = _soft_nonbonded_force(scaled)
    scaled_selector_aromatics = 0

    for atom in mol.GetAtoms():
        atom_idx = int(atom.GetIdx())
        baseline_sigma = float(baseline_force.getParticleParameters(atom_idx)[0])
        scaled_sigma = float(scaled_force.getParticleParameters(atom_idx)[0])
        is_selector_aromatic_heavy = (
            atom.GetAtomicNum() > 1
            and atom.GetIsAromatic()
            and atom.HasProp("_poly_csp_manifest_source")
            and atom.GetProp("_poly_csp_manifest_source") == "selector"
        )
        if is_selector_aromatic_heavy:
            scaled_selector_aromatics += 1
            assert abs(scaled_sigma - (baseline_sigma * 1.5)) < 1e-12
        else:
            assert abs(scaled_sigma - baseline_sigma) < 1e-12

    assert scaled_selector_aromatics > 0
    assert (
        scaled.exception_summary["scaled_aromatic_selector_atoms"]
        == scaled_selector_aromatics
    )
    assert abs(scaled.exception_summary["anti_stacking_sigma_scale"] - 1.5) < 1e-12


def test_create_system_rejects_missing_selector_payloads_early() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = _build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    glycam = _fake_glycam_params(mol)

    try:
        create_system(
            mol,
            glycam_params=glycam,
            selector_params_by_name={},
            connector_params_by_key={},
            nonbonded_mode="full",
        )
    except ValueError as exc:
        assert "No selector GAFF payload" in str(exc)
    else:
        raise AssertionError("Expected missing selector payloads to fail.")


def test_create_system_rejects_missing_connector_payloads_early() -> None:
    selector = SelectorRegistry.get("35dmpc")
    mol = _build_forcefield_mol(polymer="amylose", dp=1, selector=selector, site="C6")
    glycam = _fake_glycam_params(mol)
    selector_params = _fake_selector_params(mol, selector.name)

    try:
        create_system(
            mol,
            glycam_params=glycam,
            selector_params_by_name={selector.name: selector_params},
            connector_params_by_key={},
            nonbonded_mode="full",
        )
    except ValueError as exc:
        assert "No connector payload" in str(exc)
    else:
        raise AssertionError("Expected missing connector payloads to fail.")


def test_create_system_supports_periodic_forcefield_molecule() -> None:
    mol = _build_forcefield_mol(polymer="amylose", dp=4, end_mode="periodic")
    glycam = _fake_glycam_params(mol)

    result = create_system(mol, glycam_params=glycam, nonbonded_mode="full")
    nonbonded = next(
        force for force in (result.system.getForce(i) for i in range(result.system.getNumForces()))
        if isinstance(force, mm.NonbondedForce)
    )
    bond_force = next(
        force for force in (result.system.getForce(i) for i in range(result.system.getNumForces()))
        if isinstance(force, mm.HarmonicBondForce)
    )

    assert nonbonded.getNonbondedMethod() == mm.NonbondedForce.CutoffPeriodic
    assert result.exception_summary["periodic"] is True
    assert result.exception_summary["cutoff_nm"] > 0.0
    assert bond_force.usesPeriodicBoundaryConditions() is True
    assert all(
        entry.get("glycam_residue_role") == "periodic"
        for entry in result.topology_manifest
        if "glycam_residue_role" in entry
    )


def test_create_system_rejects_capped_forcefield_molecule_early() -> None:
    mol = _build_forcefield_mol(
        polymer="amylose",
        dp=2,
        end_mode="capped",
        end_caps={"left": "acetyl", "right": "methoxy"},
    )
    glycam = _fake_glycam_params(mol)

    try:
        create_system(mol, glycam_params=glycam, nonbonded_mode="soft")
    except ValueError as exc:
        assert "open and periodic" in str(exc)
    else:
        raise AssertionError("Expected capped runtime build to fail.")


def test_create_system_rejects_periodic_forcefield_molecule_without_box() -> None:
    mol = _build_forcefield_mol(polymer="amylose", dp=4, end_mode="periodic")
    for key in (
        "_poly_csp_box_a_A",
        "_poly_csp_box_b_A",
        "_poly_csp_box_c_A",
    ):
        mol.ClearProp(key)
    glycam = _fake_glycam_params(mol)

    try:
        create_system(mol, glycam_params=glycam, nonbonded_mode="soft")
    except ValueError as exc:
        assert "box vectors" in str(exc)
    else:
        raise AssertionError("Expected periodic runtime build without box vectors to fail.")
