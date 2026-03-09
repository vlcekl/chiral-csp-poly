"""Periodic boundary condition utilities for helical polymers.

Functions
---------
compute_helical_box_vectors
    Derive orthorhombic box dimensions from helix geometry and structure.
ensure_periodic_box_vectors
    Recompute and store periodic box vectors on the current molecule geometry.
get_box_vectors_nm
    Retrieve stored box vectors from an RDKit Mol as OpenMM Vec3 triples.
set_box_vectors
    Store box dimensions as RDKit molecule properties.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from poly_csp.config.schema import HelixSpec

# Property keys for box dimensions (stored in Angstroms).
_BOX_A_KEY = "_poly_csp_box_a_A"
_BOX_B_KEY = "_poly_csp_box_b_A"
_BOX_C_KEY = "_poly_csp_box_c_A"


def compute_helical_box_vectors(
    mol: Chem.Mol,
    helix: "HelixSpec",
    dp: int,
    padding_A: float = 30.0,
) -> tuple[float, float, float]:
    """Compute orthorhombic box vectors (Lx, Ly, Lz) in Angstroms.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule with a conformer (coordinates in Angstroms).
    helix : HelixSpec
        Helix specification (rise_A, pitch_A, etc.).
    dp : int
        Degree of polymerization.
    padding_A : float
        Padding added to the transverse dimensions to avoid self-interaction.

    Returns
    -------
    (Lx, Ly, Lz) in Angstroms.
    """
    # Axial dimension: dp * rise per residue.
    Lz = float(dp) * float(helix.rise_A)

    # Transverse dimensions: adaptive based on actual structure extent.
    if mol.GetNumConformers() > 0:
        xyz = np.asarray(
            mol.GetConformer(0).GetPositions(), dtype=float
        ).reshape((-1, 3))
        x_range = float(xyz[:, 0].max() - xyz[:, 0].min())
        y_range = float(xyz[:, 1].max() - xyz[:, 1].min())
        Lx = max(100.0, x_range + padding_A)
        Ly = max(100.0, y_range + padding_A)
    else:
        Lx = Ly = 100.0

    return Lx, Ly, Lz


def ensure_periodic_box_vectors(
    mol: Chem.Mol,
    helix: "HelixSpec",
    dp: int,
    padding_A: float = 30.0,
) -> tuple[float, float, float]:
    """Recompute and store periodic box vectors on *mol*.

    The axial box length remains tied to the commensurate helical repeat
    (`dp * rise_A`), while the transverse dimensions are refreshed from the
    current selector-bearing coordinates.
    """
    Lx_A, Ly_A, Lz_A = compute_helical_box_vectors(
        mol=mol,
        helix=helix,
        dp=dp,
        padding_A=padding_A,
    )
    set_box_vectors(mol, Lx_A, Ly_A, Lz_A)
    return Lx_A, Ly_A, Lz_A


def set_box_vectors(mol: Chem.Mol, Lx_A: float, Ly_A: float, Lz_A: float) -> None:
    """Store box dimensions (Angstroms) as molecule properties."""
    mol.SetDoubleProp(_BOX_A_KEY, float(Lx_A))
    mol.SetDoubleProp(_BOX_B_KEY, float(Ly_A))
    mol.SetDoubleProp(_BOX_C_KEY, float(Lz_A))


def get_box_vectors_A(mol: Chem.Mol) -> tuple[float, float, float] | None:
    """Retrieve box vectors (Angstroms) from molecule properties.

    Returns None if box vectors are not set.
    """
    if not (
        mol.HasProp(_BOX_A_KEY)
        and mol.HasProp(_BOX_B_KEY)
        and mol.HasProp(_BOX_C_KEY)
    ):
        return None
    return (
        float(mol.GetDoubleProp(_BOX_A_KEY)),
        float(mol.GetDoubleProp(_BOX_B_KEY)),
        float(mol.GetDoubleProp(_BOX_C_KEY)),
    )


def get_box_vectors_nm(mol: Chem.Mol):
    """Retrieve box vectors as OpenMM Vec3 triples in nanometers.

    Returns None if box vectors are not set.  Import is deferred
    so this module can be loaded without OpenMM.
    """
    box_A = get_box_vectors_A(mol)
    if box_A is None:
        return None

    from openmm import Vec3, unit  # deferred to keep module lightweight

    Lx, Ly, Lz = [v / 10.0 for v in box_A]  # Å → nm
    return (
        Vec3(Lx, 0.0, 0.0) * unit.nanometer,
        Vec3(0.0, Ly, 0.0) * unit.nanometer,
        Vec3(0.0, 0.0, Lz) * unit.nanometer,
    )
