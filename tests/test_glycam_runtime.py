from __future__ import annotations

import shutil

import pytest

openmm = pytest.importorskip("openmm")
from openmm import app as mmapp
from openmm import unit

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.glycam import (
    build_linkage_frcmod,
    build_tleap_script,
    load_glycam_params,
    run_tleap_assembly,
)
from poly_csp.forcefield.glycam_mapping import map_backbone_to_glycam
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.system_builder import build_backbone_glycam_system
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.structure.pbc import compute_helical_box_vectors, set_box_vectors
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.terminals import apply_terminal_mode


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("tleap") is None, reason="AmberTools/tleap not available"),
]


def _helix() -> HelixSpec:
    return HelixSpec(
        name="phase2_test_helix",
        theta_rad=-4.71238898038469,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=4.933333333333334,
        handedness="left",
    )


def _forcefield_backbone_mol(polymer: str, dp: int, *, end_mode: str = "open"):
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
        caps={},
        representation="anhydro",
    )
    structure = build_backbone_structure(topology, _helix()).mol
    if end_mode == "periodic":
        Lx_A, Ly_A, Lz_A = compute_helical_box_vectors(
            structure,
            _helix(),
            dp=dp,
            padding_A=30.0,
        )
        set_box_vectors(structure, Lx_A, Ly_A, Lz_A)
    return build_forcefield_molecule(structure).mol


def _reference_total_charge(polymer: str, dp: int, work_dir) -> float:
    work_dir.mkdir(parents=True, exist_ok=True)
    linkage_frcmod = build_linkage_frcmod(work_dir, filename=f"dp{dp}.frcmod")
    script = build_tleap_script(
        polymer=polymer,
        dp=dp,
        end_mode="open",
        linkage_frcmod_path=str(linkage_frcmod.resolve()),
        model_name="reference",
    )
    result = run_tleap_assembly(script, outdir=work_dir, model_name="reference")
    prmtop = mmapp.AmberPrmtopFile(str(result["prmtop"]))
    system = prmtop.createSystem()
    nonbonded = next(
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), openmm.NonbondedForce)
    )
    total_charge = 0.0
    for particle_idx in range(nonbonded.getNumParticles()):
        charge, _, _ = nonbonded.getParticleParameters(particle_idx)
        total_charge += float(charge.value_in_unit(unit.elementary_charge))
    return total_charge


def test_load_glycam_params_extracts_residue_roles_and_hydroxyl_names(tmp_path) -> None:
    params = load_glycam_params(
        polymer="amylose",
        representation="anhydro",
        end_mode="open",
        work_dir=tmp_path / "amylose_ref",
    )

    assert set(params.residue_templates) == {
        "terminal_reducing",
        "internal",
        "terminal_nonreducing",
    }
    assert "H2O" in params.residue_templates["internal"].atom_names
    assert "H3O" in params.residue_templates["internal"].atom_names
    assert "H6O" in params.residue_templates["internal"].atom_names
    assert "H4O" not in params.residue_templates["internal"].atom_names
    assert "H4O" in params.residue_templates["terminal_nonreducing"].atom_names
    assert ("terminal_reducing", "terminal_nonreducing") in params.linkage_templates


def test_load_glycam_params_extracts_periodic_role_templates(tmp_path) -> None:
    params = load_glycam_params(
        polymer="amylose",
        representation="anhydro",
        end_mode="periodic",
        work_dir=tmp_path / "amylose_periodic_ref",
    )

    assert set(params.residue_templates) == {"periodic"}
    assert set(params.linkage_templates) == {("periodic", "periodic")}
    assert params.residue_templates["periodic"].residue_name == "4GA"


def test_build_backbone_glycam_system_builds_real_nonbonded_system(tmp_path) -> None:
    mol = _forcefield_backbone_mol("amylose", dp=3)
    params = load_glycam_params(
        polymer="amylose",
        representation="anhydro",
        end_mode="open",
        work_dir=tmp_path / "amylose_system",
    )

    result = build_backbone_glycam_system(mol, params)

    assert result.system.getNumParticles() == mol.GetNumAtoms()
    assert result.nonbonded_mode == "full"
    assert len(result.topology_manifest) == mol.GetNumAtoms()
    assert result.exception_summary["exceptions_seen"] > 0

    forces = [result.system.getForce(i) for i in range(result.system.getNumForces())]
    assert any(isinstance(force, openmm.NonbondedForce) for force in forces)
    assert not any(isinstance(force, openmm.CustomNonbondedForce) for force in forces)

    nonbonded = next(force for force in forces if isinstance(force, openmm.NonbondedForce))
    total_charge = 0.0
    for particle_idx in range(nonbonded.getNumParticles()):
        charge, _, _ = nonbonded.getParticleParameters(particle_idx)
        total_charge += float(charge.value_in_unit(unit.elementary_charge))
    expected_charge = _reference_total_charge(
        "amylose",
        dp=3,
        work_dir=tmp_path / "amylose_charge_ref",
    )
    assert abs(total_charge - expected_charge) < 1e-6


def test_build_backbone_glycam_system_supports_cellulose(tmp_path) -> None:
    mol = _forcefield_backbone_mol("cellulose", dp=2)
    params = load_glycam_params(
        polymer="cellulose",
        representation="anhydro",
        end_mode="open",
        work_dir=tmp_path / "cellulose_system",
    )

    result = build_backbone_glycam_system(mol, params)
    residue_names = {entry["glycam_residue_name"] for entry in result.topology_manifest}
    assert residue_names == {"4GB", "0GB"}
    assert result.system.getNumParticles() == mol.GetNumAtoms()


def test_build_backbone_glycam_system_supports_periodic_backbone(tmp_path) -> None:
    mol = _forcefield_backbone_mol("amylose", dp=4, end_mode="periodic")
    params = load_glycam_params(
        polymer="amylose",
        representation="anhydro",
        end_mode="periodic",
        work_dir=tmp_path / "amylose_periodic_system",
    )

    result = build_backbone_glycam_system(mol, params)
    nonbonded = next(
        force for force in (result.system.getForce(i) for i in range(result.system.getNumForces()))
        if isinstance(force, openmm.NonbondedForce)
    )

    assert all(
        entry["glycam_residue_role"] == "periodic"
        for entry in result.topology_manifest
        if "glycam_residue_role" in entry
    )
    assert nonbonded.getNonbondedMethod() == openmm.NonbondedForce.CutoffPeriodic
    assert result.exception_summary["periodic"] is True


def test_map_backbone_to_glycam_ignores_selector_atoms(tmp_path) -> None:
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
        selector=SelectorRegistry.get("35dmpc"),
    )
    mol = build_forcefield_molecule(structure).mol
    params = load_glycam_params(
        polymer="amylose",
        representation="anhydro",
        end_mode="open",
        work_dir=tmp_path / "selector_reject",
    )

    mapping = map_backbone_to_glycam(mol, params)
    assert mapping.assignments
    assert all(assignment.generic_atom_name.startswith(("C", "O", "H")) for assignment in mapping.assignments)
    assert all(
        mol.GetAtomWithIdx(assignment.atom_index).GetProp("_poly_csp_manifest_source") == "backbone"
        for assignment in mapping.assignments
    )


def test_load_glycam_params_rejects_unsupported_representation() -> None:
    with pytest.raises(ValueError, match="anhydro"):
        load_glycam_params(
            polymer="amylose",
            representation="natural_oh",
            end_mode="open",
        )
