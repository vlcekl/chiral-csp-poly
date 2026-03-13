from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem

import openmm as mm

from poly_csp.config.schema import SoftSelectorHbondBiasOptions


@dataclass(frozen=True)
class _SelectorBiasAtomRecord:
    residue_index: int
    instance_id: int
    atom_idx: int


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _periodic_residue_gap(
    left: int,
    right: int,
    *,
    dp: int,
    periodic: bool,
) -> int:
    diff = abs(int(left) - int(right))
    if not periodic or dp <= 1:
        return diff
    return min(diff, int(dp) - diff)


def _selector_bias_atom_records(
    mol: Chem.Mol,
    *,
    connector_role: str,
    atomic_num: int,
) -> list[_SelectorBiasAtomRecord]:
    out: list[_SelectorBiasAtomRecord] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != int(atomic_num):
            continue
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_connector_role"):
            continue
        if str(atom.GetProp("_poly_csp_connector_role")) != connector_role:
            continue
        out.append(
            _SelectorBiasAtomRecord(
                residue_index=int(atom.GetIntProp("_poly_csp_residue_index")),
                instance_id=int(atom.GetIntProp("_poly_csp_selector_instance")),
                atom_idx=int(atom.GetIdx()),
            )
        )
    return out


def selector_soft_hbond_bias_pairs(
    mol: Chem.Mol,
    *,
    neighbor_window: int = 1,
    excluded_pairs: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Return static selector carbamate H...O bias pairs from topology metadata."""
    donor_h = _selector_bias_atom_records(
        mol,
        connector_role="amide_n",
        atomic_num=1,
    )
    acceptor_o = _selector_bias_atom_records(
        mol,
        connector_role="carbonyl_o",
        atomic_num=8,
    )
    if not donor_h or not acceptor_o:
        return []

    periodic = bool(
        mol.HasProp("_poly_csp_end_mode")
        and str(mol.GetProp("_poly_csp_end_mode")).strip().lower() == "periodic"
    )
    dp = int(mol.GetIntProp("_poly_csp_dp")) if mol.HasProp("_poly_csp_dp") else 0
    excluded = {_bond_key(int(i), int(j)) for i, j in excluded_pairs}
    pairs: list[tuple[int, int]] = []
    for donor in donor_h:
        for acceptor in acceptor_o:
            if donor.instance_id == acceptor.instance_id:
                continue
            if _periodic_residue_gap(
                donor.residue_index,
                acceptor.residue_index,
                dp=dp,
                periodic=periodic,
            ) > int(neighbor_window):
                continue
            pair = _bond_key(donor.atom_idx, acceptor.atom_idx)
            if pair in excluded:
                continue
            pairs.append((int(donor.atom_idx), int(acceptor.atom_idx)))
    return sorted(set(pairs))


def add_soft_selector_hbond_bias_force(
    system: mm.System,
    mol: Chem.Mol,
    *,
    options: SoftSelectorHbondBiasOptions,
    periodic: bool = False,
    excluded_pairs: Iterable[tuple[int, int]] = (),
) -> dict[str, object]:
    summary: dict[str, object] = {
        "enabled": bool(options.enabled),
        "force_kind": None,
        "pair_count": 0,
        "epsilon_kj_per_mol": float(options.epsilon_kj_per_mol),
        "r0_nm": float(options.r0_nm),
        "half_width_nm": float(options.half_width_nm),
        "window_lower_nm": float(options.r0_nm - options.half_width_nm),
        "window_upper_nm": float(options.r0_nm + options.half_width_nm),
        "hbond_neighbor_window": int(options.hbond_neighbor_window),
    }
    if not bool(options.enabled):
        return summary

    pairs = selector_soft_hbond_bias_pairs(
        mol,
        neighbor_window=int(options.hbond_neighbor_window),
        excluded_pairs=excluded_pairs,
    )
    summary["pair_count"] = len(pairs)
    if not pairs:
        return summary

    force = mm.CustomBondForce(
        "-k_soft_hb_scale*epsilon*step(1-u*u)*(1-u*u)^2; "
        "u=(r-r0)/half_width"
    )
    force.setUsesPeriodicBoundaryConditions(bool(periodic))
    force.addGlobalParameter("k_soft_hb_scale", 1.0)
    force.addPerBondParameter("epsilon")
    force.addPerBondParameter("r0")
    force.addPerBondParameter("half_width")
    epsilon = float(options.epsilon_kj_per_mol)
    r0 = float(options.r0_nm)
    half_width = float(options.half_width_nm)
    for atom_a, atom_b in pairs:
        force.addBond(int(atom_a), int(atom_b), [epsilon, r0, half_width])
    system.addForce(force)
    summary["force_kind"] = "CustomBondForce"
    return summary
