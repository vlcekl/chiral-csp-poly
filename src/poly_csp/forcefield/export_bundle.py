from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rdkit import Chem

import openmm as mm
from openmm import app as mmapp, unit

from poly_csp.forcefield.runtime_params import RuntimeParams
from poly_csp.forcefield.system_builder import SystemBuildResult, create_system
from poly_csp.structure.pbc import get_box_vectors_nm


@dataclass(frozen=True)
class NonbondedParticleParams:
    charge_e: float
    sigma_nm: float
    epsilon_kj_per_mol: float


@dataclass(frozen=True)
class ExportBundle:
    mol: Chem.Mol
    system_build: SystemBuildResult
    topology: mmapp.Topology
    positions_nm: unit.Quantity
    nonbonded_particles: tuple[NonbondedParticleParams, ...]
    box_vectors_nm: tuple[object, object, object] | None = None


def _atom_name(atom: Chem.Atom) -> str:
    if atom.HasProp("_poly_csp_atom_name"):
        return atom.GetProp("_poly_csp_atom_name")
    info = atom.GetPDBResidueInfo()
    if info is not None:
        name = info.GetName().strip()
        if name:
            return name
    return f"{atom.GetSymbol()}{atom.GetIdx() + 1}"


def _residue_identity(atom: Chem.Atom) -> tuple[str, str, int]:
    info = atom.GetPDBResidueInfo()
    if info is not None:
        chain_id = info.GetChainId().strip() or "A"
        residue_name = info.GetResidueName().strip() or "MOL"
        residue_number = int(info.GetResidueNumber()) if info.GetResidueNumber() else 1
        return chain_id, residue_name, residue_number

    if atom.HasProp("_poly_csp_selector_instance"):
        return "B", "SEL", int(atom.GetIntProp("_poly_csp_selector_instance"))

    if atom.HasProp("_poly_csp_residue_index"):
        return "A", "GLC", int(atom.GetIntProp("_poly_csp_residue_index")) + 1

    return "A", "MOL", 1


def build_openmm_topology_from_mol(mol: Chem.Mol) -> mmapp.Topology:
    topology = mmapp.Topology()
    atom_refs: list[mmapp.Atom] = [None] * mol.GetNumAtoms()  # type: ignore[list-item]
    chain_refs: dict[str, mmapp.Chain] = {}
    current_residue = None
    current_residue_key: tuple[str, str, int] | None = None

    for atom in mol.GetAtoms():
        chain_id, residue_name, residue_number = _residue_identity(atom)
        chain = chain_refs.get(chain_id)
        if chain is None:
            chain = topology.addChain(chain_id)
            chain_refs[chain_id] = chain

        residue_key = (chain_id, residue_name, residue_number)
        if current_residue is None or residue_key != current_residue_key:
            current_residue = topology.addResidue(
                residue_name,
                chain,
                id=str(residue_number),
            )
            current_residue_key = residue_key

        element = None
        if atom.GetAtomicNum() > 0:
            element = mmapp.Element.getByAtomicNumber(atom.GetAtomicNum())
        atom_refs[atom.GetIdx()] = topology.addAtom(
            _atom_name(atom),
            element,
            current_residue,
        )

    for bond in mol.GetBonds():
        topology.addBond(
            atom_refs[bond.GetBeginAtomIdx()],
            atom_refs[bond.GetEndAtomIdx()],
            order=int(round(bond.GetBondTypeAsDouble())),
        )

    box_vectors_nm = get_box_vectors_nm(mol)
    if box_vectors_nm is not None:
        topology.setPeriodicBoxVectors(box_vectors_nm)
    return topology


def extract_nonbonded_particles(
    system: mm.System,
) -> tuple[NonbondedParticleParams, ...]:
    nonbonded = None
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, mm.NonbondedForce):
            if nonbonded is not None:
                raise ValueError("Expected exactly one NonbondedForce in the full runtime system.")
            nonbonded = force
    if nonbonded is None:
        raise ValueError(
            "Canonical export requires a full runtime system with a NonbondedForce."
        )

    particles: list[NonbondedParticleParams] = []
    for particle_index in range(nonbonded.getNumParticles()):
        charge, sigma, epsilon = nonbonded.getParticleParameters(particle_index)
        particles.append(
            NonbondedParticleParams(
                charge_e=float(charge.value_in_unit(unit.elementary_charge)),
                sigma_nm=float(sigma.value_in_unit(unit.nanometer)),
                epsilon_kj_per_mol=float(
                    epsilon.value_in_unit(unit.kilojoule_per_mole)
                ),
            )
        )
    return tuple(particles)


def prepare_export_bundle(
    mol: Chem.Mol,
    *,
    runtime_params: RuntimeParams | None = None,
    system_build: SystemBuildResult | None = None,
    mixing_rules_cfg: Mapping[str, object] | None = None,
    repulsion_k_kj_per_mol_nm2: float = 800.0,
    repulsion_cutoff_nm: float = 0.6,
) -> ExportBundle:
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Canonical export requires a forcefield-domain molecule from build_forcefield_molecule()."
        )

    resolved_system = system_build
    if resolved_system is None:
        if runtime_params is None:
            raise ValueError(
                "prepare_export_bundle() requires runtime_params when no prebuilt full system is supplied."
            )
        resolved_system = create_system(
            mol,
            glycam_params=runtime_params.glycam,
            selector_params_by_name=runtime_params.selector_params_by_name,
            connector_params_by_key=runtime_params.connector_params_by_key,
            parameter_provenance=runtime_params.source_manifest,
            nonbonded_mode="full",
            mixing_rules_cfg=mixing_rules_cfg,
            repulsion_k_kj_per_mol_nm2=float(repulsion_k_kj_per_mol_nm2),
            repulsion_cutoff_nm=float(repulsion_cutoff_nm),
        )

    if resolved_system.nonbonded_mode != "full":
        raise ValueError(
            "Canonical export requires a full runtime system; got "
            f"{resolved_system.nonbonded_mode!r}."
        )
    if resolved_system.system.getNumParticles() != mol.GetNumAtoms():
        raise ValueError(
            "Canonical export bundle atom-count mismatch between molecule and system: "
            f"mol={mol.GetNumAtoms()}, system={resolved_system.system.getNumParticles()}."
        )

    topology = build_openmm_topology_from_mol(mol)
    nonbonded_particles = extract_nonbonded_particles(resolved_system.system)
    if len(nonbonded_particles) != mol.GetNumAtoms():
        raise ValueError(
            "Canonical export bundle particle-count mismatch between molecule and NonbondedForce: "
            f"mol={mol.GetNumAtoms()}, nonbonded={len(nonbonded_particles)}."
        )

    return ExportBundle(
        mol=mol,
        system_build=resolved_system,
        topology=topology,
        positions_nm=resolved_system.positions_nm,
        nonbonded_particles=nonbonded_particles,
        box_vectors_nm=get_box_vectors_nm(mol),
    )
