"""Deterministic mapping from forcefield-domain selector instances to GAFF payloads."""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from poly_csp.forcefield.gaff import SelectorFragmentParams


@dataclass(frozen=True)
class SelectorInstanceMapping:
    instance_id: int
    selector_name: str
    atom_index_by_name: dict[str, int]


def map_selector_instances(
    mol: Chem.Mol,
    selector_params_by_name: dict[str, SelectorFragmentParams],
) -> dict[int, SelectorInstanceMapping]:
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Selector mapping requires a forcefield-domain molecule from build_forcefield_molecule()."
        )

    by_instance: dict[int, dict[str, object]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_manifest_source"):
            raise ValueError(
                f"Atom {atom.GetIdx()} is missing _poly_csp_manifest_source; expected forcefield manifest metadata."
            )
        if atom.GetProp("_poly_csp_manifest_source") != "selector":
            continue
        if not atom.HasProp("_poly_csp_selector_instance"):
            raise ValueError(
                f"Selector atom {atom.GetIdx()} is missing _poly_csp_selector_instance."
            )
        if not atom.HasProp("_poly_csp_selector_name"):
            raise ValueError(
                f"Selector atom {atom.GetIdx()} is missing _poly_csp_selector_name."
            )
        if not atom.HasProp("_poly_csp_atom_name"):
            raise ValueError(
                f"Selector atom {atom.GetIdx()} is missing _poly_csp_atom_name."
            )

        instance_id = int(atom.GetIntProp("_poly_csp_selector_instance"))
        selector_name = atom.GetProp("_poly_csp_selector_name")
        entry = by_instance.setdefault(
            instance_id,
            {"selector_name": selector_name, "atom_index_by_name": {}},
        )
        if entry["selector_name"] != selector_name:
            raise ValueError(
                f"Selector instance {instance_id} mixes selector names "
                f"{entry['selector_name']!r} and {selector_name!r}."
            )
        atom_name = atom.GetProp("_poly_csp_atom_name")
        entry["atom_index_by_name"][atom_name] = int(atom.GetIdx())  # type: ignore[index]

    out: dict[int, SelectorInstanceMapping] = {}
    for instance_id, payload in by_instance.items():
        selector_name = str(payload["selector_name"])
        params = selector_params_by_name.get(selector_name)
        if params is None:
            raise ValueError(
                f"No selector GAFF payload is available for selector {selector_name!r}."
            )
        atom_index_by_name = dict(payload["atom_index_by_name"])  # type: ignore[arg-type]
        observed = set(atom_index_by_name)
        expected = set(params.atom_params)
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extra = sorted(observed.difference(expected))
            raise ValueError(
                "Selector instance atom-set mismatch for "
                f"instance {instance_id} ({selector_name}). Missing={missing}, extra={extra}."
            )
        out[instance_id] = SelectorInstanceMapping(
            instance_id=instance_id,
            selector_name=selector_name,
            atom_index_by_name=atom_index_by_name,
        )
    return out

