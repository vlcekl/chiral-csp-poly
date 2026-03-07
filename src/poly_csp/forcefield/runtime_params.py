"""Canonical runtime parameter loading for the supported forcefield slice."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from poly_csp.forcefield.connectors import ConnectorParams, load_connector_params
from poly_csp.forcefield.gaff import SelectorFragmentParams, load_selector_fragment_params
from poly_csp.forcefield.glycam import GlycamParams, load_glycam_params


@dataclass(frozen=True)
class RuntimeParams:
    glycam: GlycamParams
    selector_params_by_name: dict[str, SelectorFragmentParams]
    connector_params_by_key: dict[tuple[str, str], ConnectorParams]


def load_runtime_params(
    mol: Chem.Mol,
    selector_template=None,
    work_dir: Path | None = None,
) -> RuntimeParams:
    if not mol.HasProp("_poly_csp_polymer"):
        raise ValueError("Forcefield-domain molecule is missing _poly_csp_polymer.")
    if not mol.HasProp("_poly_csp_representation"):
        raise ValueError("Forcefield-domain molecule is missing _poly_csp_representation.")
    if not mol.HasProp("_poly_csp_end_mode"):
        raise ValueError("Forcefield-domain molecule is missing _poly_csp_end_mode.")

    polymer = mol.GetProp("_poly_csp_polymer")
    representation = mol.GetProp("_poly_csp_representation")
    end_mode = mol.GetProp("_poly_csp_end_mode")

    glycam = load_glycam_params(
        polymer=polymer,  # type: ignore[arg-type]
        representation=representation,  # type: ignore[arg-type]
        end_mode=end_mode,  # type: ignore[arg-type]
        work_dir=None if work_dir is None else work_dir / "glycam",
    )

    selector_params_by_name: dict[str, SelectorFragmentParams] = {}
    connector_params_by_key: dict[tuple[str, str], ConnectorParams] = {}

    selector_instance_atoms = [
        atom for atom in mol.GetAtoms() if atom.HasProp("_poly_csp_selector_instance")
    ]
    if not selector_instance_atoms:
        return RuntimeParams(
            glycam=glycam,
            selector_params_by_name=selector_params_by_name,
            connector_params_by_key=connector_params_by_key,
        )

    if selector_template is None:
        raise ValueError(
            "Selector-bearing runtime parameter loading requires the SelectorTemplate."
        )

    selector_params_by_name[selector_template.name] = load_selector_fragment_params(
        selector_template=selector_template,
        work_dir=None if work_dir is None else work_dir / "selector",
    )

    sites = sorted(
        {
            atom.GetProp("_poly_csp_site")
            for atom in selector_instance_atoms
            if atom.HasProp("_poly_csp_site")
        }
    )
    for site in sites:
        connector_params_by_key[(selector_template.name, site)] = load_connector_params(
            polymer=polymer,  # type: ignore[arg-type]
            selector_template=selector_template,
            site=site,  # type: ignore[arg-type]
            monomer_representation="natural_oh",
            work_dir=None if work_dir is None else work_dir / f"connector_{site.lower()}",
        )

    return RuntimeParams(
        glycam=glycam,
        selector_params_by_name=selector_params_by_name,
        connector_params_by_key=connector_params_by_key,
    )

