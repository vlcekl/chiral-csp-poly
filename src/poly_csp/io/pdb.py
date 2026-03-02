# poly_csp/io/pdb.py
"""PDB output with residue/chain annotations from polymer metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import rdchem


def _assign_pdb_info(mol: Chem.Mol) -> None:
    """Set PDBResidueInfo on each atom using poly_csp metadata.

    - Backbone atoms: chain A, residue name "GLC", residue number from index.
    - Selector atoms: chain B, residue name "SEL", residue number from index.
    """
    # Build per-atom residue index lookup from label maps
    residue_for_atom: Dict[int, int] = {}
    if mol.HasProp("_poly_csp_residue_label_map_json"):
        maps = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
        for res_idx, mapping in enumerate(maps):
            for label, atom_idx in mapping.items():
                residue_for_atom[int(atom_idx)] = res_idx

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        is_selector = atom.HasProp("_poly_csp_selector_instance")

        if is_selector:
            res_name = "SEL"
            chain_id = "B"
            res_idx = (
                int(atom.GetIntProp("_poly_csp_residue_index"))
                if atom.HasProp("_poly_csp_residue_index")
                else 0
            )
            local_idx = (
                int(atom.GetIntProp("_poly_csp_selector_local_idx"))
                if atom.HasProp("_poly_csp_selector_local_idx")
                else idx
            )
            atom_name = f"{atom.GetSymbol()}{local_idx}"
        else:
            res_name = "GLC"
            chain_id = "A"
            res_idx = residue_for_atom.get(idx, 0)
            # Try to get atom label from the residue label maps
            atom_name = atom.GetSymbol() + str(idx)
            if mol.HasProp("_poly_csp_residue_label_map_json"):
                maps = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
                if res_idx < len(maps):
                    for label, a_idx in maps[res_idx].items():
                        if int(a_idx) == idx:
                            atom_name = label
                            break

        # Format atom name: PDB convention is 4 chars, left-justified for
        # 2-char element symbols, otherwise right-padded.
        if len(atom_name) < 4:
            atom_name = f" {atom_name:<3s}"
        elif len(atom_name) > 4:
            atom_name = atom_name[:4]

        info = rdchem.AtomPDBResidueInfo()
        info.SetName(atom_name)
        info.SetResidueName(res_name)
        info.SetResidueNumber(res_idx + 1)  # PDB uses 1-based
        info.SetChainId(chain_id)
        info.SetIsHeteroAtom(is_selector)
        info.SetOccupancy(1.0)
        info.SetTempFactor(0.0)
        atom.SetPDBResidueInfo(info)


def write_pdb_from_rdkit(mol: Chem.Mol, path: str | Path) -> None:
    """Write PDB with residue/chain annotations from polymer metadata."""
    path = Path(path)

    # Work on a copy so we don't mutate the original
    out = Chem.Mol(mol)

    # Assign PDB info if polymer metadata is available
    has_metadata = (
        out.HasProp("_poly_csp_residue_label_map_json")
        or any(a.HasProp("_poly_csp_selector_instance") for a in out.GetAtoms())
    )
    if has_metadata:
        _assign_pdb_info(out)

    pdb = Chem.MolToPDBBlock(out)
    path.write_text(pdb, encoding="utf-8")