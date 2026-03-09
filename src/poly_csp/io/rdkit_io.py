from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from poly_csp.topology.monomers import (
    MonomerRepresentation,
    PolymerKind,
    make_glucose_template,
)


def write_mol(mol: Chem.Mol, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(mol, str(path))


def write_sdf(mol: Chem.Mol, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    if writer is None:
        raise RuntimeError(f"Could not open SDF writer for {path}")
    writer.write(mol)
    writer.close()


def export_glucose_template_sdf(
    polymer: PolymerKind,
    path: str | Path,
    monomer_representation: MonomerRepresentation = "anhydro",
) -> None:
    template = make_glucose_template(polymer, monomer_representation=monomer_representation)
    write_sdf(template.mol, path)
