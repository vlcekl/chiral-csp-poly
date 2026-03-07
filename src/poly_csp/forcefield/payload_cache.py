from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rdkit import Chem

from poly_csp.config.schema import EndMode, MonomerRepresentation, PolymerKind, Site
from poly_csp.forcefield.connectors import (
    ConnectorAngleTemplate,
    ConnectorAtomParams,
    ConnectorBondTemplate,
    ConnectorParams,
    ConnectorToken,
    ConnectorTorsionTemplate,
)
from poly_csp.forcefield.gaff import (
    SelectorAngleTemplate,
    SelectorAtomParams,
    SelectorBondTemplate,
    SelectorFragmentParams,
    SelectorTorsionTemplate,
)
from poly_csp.topology.selectors import SelectorTemplate

if TYPE_CHECKING:
    from poly_csp.forcefield.glycam import (
        GlycamAngleTemplate,
        GlycamAtomParams,
        GlycamAtomToken,
        GlycamBondTemplate,
        GlycamLinkageTemplate,
        GlycamParams,
        GlycamResidueTemplate,
        GlycamTorsionTemplate,
    )


PAYLOAD_CACHE_SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CACHE_DIR = _REPO_ROOT / ".cache" / "poly_csp" / "runtime_params"


def resolve_runtime_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is None:
        return DEFAULT_RUNTIME_CACHE_DIR
    path = Path(cache_dir)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _canonical_smiles(mol: Chem.Mol) -> str:
    canonical = Chem.Mol(mol)
    for atom in canonical.GetAtoms():
        atom.SetAtomMapNum(0)
    return str(Chem.MolToSmiles(canonical, canonical=True, allHsExplicit=True))


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def selector_cache_identity(
    selector_template: SelectorTemplate,
    *,
    charge_model: str = "bcc",
    net_charge: int = 0,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": PAYLOAD_CACHE_SCHEMA_VERSION,
        "kind": "selector_fragment",
        "selector_name": str(selector_template.name),
        "selector_smiles": _canonical_smiles(selector_template.mol),
        "attach_atom_idx": int(selector_template.attach_atom_idx),
        "attach_dummy_idx": (
            int(selector_template.attach_dummy_idx)
            if selector_template.attach_dummy_idx is not None
            else None
        ),
        "linkage_type": str(selector_template.linkage_type),
        "connector_local_roles": {
            str(int(local_idx)): str(role)
            for local_idx, role in sorted(selector_template.connector_local_roles.items())
        },
        "charge_model": str(charge_model),
        "net_charge": int(net_charge),
    }
    return _stable_hash(identity), identity


def connector_cache_identity(
    polymer: PolymerKind,
    selector_template: SelectorTemplate,
    site: Site,
    *,
    monomer_representation: MonomerRepresentation = "natural_oh",
    charge_model: str = "bcc",
    net_charge: int = 0,
) -> tuple[str, dict[str, Any]]:
    selector_key, selector_identity = selector_cache_identity(
        selector_template,
        charge_model=charge_model,
        net_charge=net_charge,
    )
    identity = {
        "schema_version": PAYLOAD_CACHE_SCHEMA_VERSION,
        "kind": "connector_fragment",
        "polymer": str(polymer),
        "site": str(site),
        "monomer_representation": str(monomer_representation),
        "selector_key": selector_key,
        "selector_identity": selector_identity,
        "charge_model": str(charge_model),
        "net_charge": int(net_charge),
    }
    return _stable_hash(identity), identity


def selector_cache_dir(
    cache_root: str | Path | None,
    selector_template: SelectorTemplate,
    *,
    charge_model: str = "bcc",
    net_charge: int = 0,
) -> tuple[Path, dict[str, Any]]:
    key, identity = selector_cache_identity(
        selector_template,
        charge_model=charge_model,
        net_charge=net_charge,
    )
    root = resolve_runtime_cache_dir(cache_root)
    return root / "selector" / str(selector_template.name).lower() / key, identity


def connector_cache_dir(
    cache_root: str | Path | None,
    polymer: PolymerKind,
    selector_template: SelectorTemplate,
    site: Site,
    *,
    monomer_representation: MonomerRepresentation = "natural_oh",
    charge_model: str = "bcc",
    net_charge: int = 0,
) -> tuple[Path, dict[str, Any]]:
    key, identity = connector_cache_identity(
        polymer,
        selector_template,
        site,
        monomer_representation=monomer_representation,
        charge_model=charge_model,
        net_charge=net_charge,
    )
    root = resolve_runtime_cache_dir(cache_root)
    return (
        root / "connector" / str(selector_template.name).lower() / str(site).lower() / key,
        identity,
    )


def glycam_cache_identity(
    polymer: PolymerKind,
    representation: MonomerRepresentation,
    end_mode: EndMode,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": PAYLOAD_CACHE_SCHEMA_VERSION,
        "kind": "glycam_backbone",
        "polymer": str(polymer),
        "representation": str(representation),
        "end_mode": str(end_mode),
    }
    return _stable_hash(identity), identity


def glycam_cache_dir(
    cache_root: str | Path | None,
    polymer: PolymerKind,
    representation: MonomerRepresentation,
    end_mode: EndMode,
) -> tuple[Path, dict[str, Any]]:
    key, identity = glycam_cache_identity(
        polymer=polymer,
        representation=representation,
        end_mode=end_mode,
    )
    root = resolve_runtime_cache_dir(cache_root)
    return (
        root
        / "glycam"
        / str(polymer).lower()
        / str(representation).lower()
        / str(end_mode).lower()
        / key,
        identity,
    )


def _selector_params_to_jsonable(params: SelectorFragmentParams) -> dict[str, Any]:
    return {
        "selector_name": params.selector_name,
        "source_prmtop": params.source_prmtop,
        "fragment_atom_count": params.fragment_atom_count,
        "atom_params": {
            name: {
                "atom_name": atom.atom_name,
                "charge_e": atom.charge_e,
                "sigma_nm": atom.sigma_nm,
                "epsilon_kj_per_mol": atom.epsilon_kj_per_mol,
            }
            for name, atom in sorted(params.atom_params.items())
        },
        "bonds": [
            {
                "atom_names": list(bond.atom_names),
                "length_nm": bond.length_nm,
                "k_kj_per_mol_nm2": bond.k_kj_per_mol_nm2,
            }
            for bond in params.bonds
        ],
        "angles": [
            {
                "atom_names": list(angle.atom_names),
                "theta0_rad": angle.theta0_rad,
                "k_kj_per_mol_rad2": angle.k_kj_per_mol_rad2,
            }
            for angle in params.angles
        ],
        "torsions": [
            {
                "atom_names": list(torsion.atom_names),
                "periodicity": torsion.periodicity,
                "phase_rad": torsion.phase_rad,
                "k_kj_per_mol": torsion.k_kj_per_mol,
            }
            for torsion in params.torsions
        ],
    }


def _selector_params_from_jsonable(data: dict[str, Any]) -> SelectorFragmentParams:
    return SelectorFragmentParams(
        selector_name=str(data["selector_name"]),
        atom_params={
            str(name): SelectorAtomParams(
                atom_name=str(payload["atom_name"]),
                charge_e=float(payload["charge_e"]),
                sigma_nm=float(payload["sigma_nm"]),
                epsilon_kj_per_mol=float(payload["epsilon_kj_per_mol"]),
            )
            for name, payload in dict(data["atom_params"]).items()
        },
        bonds=tuple(
            SelectorBondTemplate(
                atom_names=tuple(str(name) for name in bond["atom_names"]),  # type: ignore[arg-type]
                length_nm=float(bond["length_nm"]),
                k_kj_per_mol_nm2=float(bond["k_kj_per_mol_nm2"]),
            )
            for bond in list(data["bonds"])
        ),
        angles=tuple(
            SelectorAngleTemplate(
                atom_names=tuple(str(name) for name in angle["atom_names"]),  # type: ignore[arg-type]
                theta0_rad=float(angle["theta0_rad"]),
                k_kj_per_mol_rad2=float(angle["k_kj_per_mol_rad2"]),
            )
            for angle in list(data["angles"])
        ),
        torsions=tuple(
            SelectorTorsionTemplate(
                atom_names=tuple(str(name) for name in torsion["atom_names"]),  # type: ignore[arg-type]
                periodicity=int(torsion["periodicity"]),
                phase_rad=float(torsion["phase_rad"]),
                k_kj_per_mol=float(torsion["k_kj_per_mol"]),
            )
            for torsion in list(data["torsions"])
        ),
        source_prmtop=(
            str(data["source_prmtop"])
            if data.get("source_prmtop") is not None
            else None
        ),
        fragment_atom_count=(
            int(data["fragment_atom_count"])
            if data.get("fragment_atom_count") is not None
            else None
        ),
    )


def _connector_token_to_jsonable(token: ConnectorToken) -> dict[str, Any]:
    return {"source": str(token.source), "atom_name": str(token.atom_name)}


def _connector_token_from_jsonable(data: dict[str, Any]) -> ConnectorToken:
    return ConnectorToken(source=str(data["source"]), atom_name=str(data["atom_name"]))


def _connector_params_to_jsonable(params: ConnectorParams) -> dict[str, Any]:
    return {
        "polymer": params.polymer,
        "selector_name": params.selector_name,
        "site": params.site,
        "monomer_representation": params.monomer_representation,
        "source_prmtop": params.source_prmtop,
        "fragment_atom_count": params.fragment_atom_count,
        "atom_params": {
            name: {
                "atom_name": atom.atom_name,
                "charge_e": atom.charge_e,
                "sigma_nm": atom.sigma_nm,
                "epsilon_kj_per_mol": atom.epsilon_kj_per_mol,
            }
            for name, atom in sorted(params.atom_params.items())
        },
        "bonds": [
            {
                "atoms": [_connector_token_to_jsonable(token) for token in bond.atoms],
                "length_nm": bond.length_nm,
                "k_kj_per_mol_nm2": bond.k_kj_per_mol_nm2,
            }
            for bond in params.bonds
        ],
        "angles": [
            {
                "atoms": [_connector_token_to_jsonable(token) for token in angle.atoms],
                "theta0_rad": angle.theta0_rad,
                "k_kj_per_mol_rad2": angle.k_kj_per_mol_rad2,
            }
            for angle in params.angles
        ],
        "torsions": [
            {
                "atoms": [_connector_token_to_jsonable(token) for token in torsion.atoms],
                "periodicity": torsion.periodicity,
                "phase_rad": torsion.phase_rad,
                "k_kj_per_mol": torsion.k_kj_per_mol,
            }
            for torsion in params.torsions
        ],
    }


def _connector_params_from_jsonable(data: dict[str, Any]) -> ConnectorParams:
    return ConnectorParams(
        polymer=str(data["polymer"]) if data.get("polymer") is not None else None,
        selector_name=(
            str(data["selector_name"]) if data.get("selector_name") is not None else None
        ),
        site=str(data["site"]) if data.get("site") is not None else None,
        monomer_representation=(
            str(data["monomer_representation"])
            if data.get("monomer_representation") is not None
            else None
        ),
        atom_params={
            str(name): ConnectorAtomParams(
                atom_name=str(payload["atom_name"]),
                charge_e=float(payload["charge_e"]),
                sigma_nm=float(payload["sigma_nm"]),
                epsilon_kj_per_mol=float(payload["epsilon_kj_per_mol"]),
            )
            for name, payload in dict(data["atom_params"]).items()
        },
        bonds=tuple(
            ConnectorBondTemplate(
                atoms=tuple(
                    _connector_token_from_jsonable(token)
                    for token in list(bond["atoms"])
                ),  # type: ignore[arg-type]
                length_nm=float(bond["length_nm"]),
                k_kj_per_mol_nm2=float(bond["k_kj_per_mol_nm2"]),
            )
            for bond in list(data["bonds"])
        ),
        angles=tuple(
            ConnectorAngleTemplate(
                atoms=tuple(
                    _connector_token_from_jsonable(token)
                    for token in list(angle["atoms"])
                ),  # type: ignore[arg-type]
                theta0_rad=float(angle["theta0_rad"]),
                k_kj_per_mol_rad2=float(angle["k_kj_per_mol_rad2"]),
            )
            for angle in list(data["angles"])
        ),
        torsions=tuple(
            ConnectorTorsionTemplate(
                atoms=tuple(
                    _connector_token_from_jsonable(token)
                    for token in list(torsion["atoms"])
                ),  # type: ignore[arg-type]
                periodicity=int(torsion["periodicity"]),
                phase_rad=float(torsion["phase_rad"]),
                k_kj_per_mol=float(torsion["k_kj_per_mol"]),
            )
            for torsion in list(data["torsions"])
        ),
        source_prmtop=(
            str(data["source_prmtop"])
            if data.get("source_prmtop") is not None
            else None
        ),
        fragment_atom_count=(
            int(data["fragment_atom_count"])
            if data.get("fragment_atom_count") is not None
            else None
        ),
    )


def _glycam_atom_token_to_jsonable(token: "GlycamAtomToken") -> dict[str, Any]:
    return {
        "residue_offset": int(token.residue_offset),
        "atom_name": str(token.atom_name),
    }


def _glycam_atom_token_from_jsonable(data: dict[str, Any]) -> "GlycamAtomToken":
    from poly_csp.forcefield.glycam import GlycamAtomToken

    return GlycamAtomToken(
        residue_offset=int(data["residue_offset"]),
        atom_name=str(data["atom_name"]),
    )


def _glycam_bond_to_jsonable(template: "GlycamBondTemplate") -> dict[str, Any]:
    return {
        "atoms": [_glycam_atom_token_to_jsonable(token) for token in template.atoms],
        "length_nm": template.length_nm,
        "k_kj_per_mol_nm2": template.k_kj_per_mol_nm2,
    }


def _glycam_angle_to_jsonable(template: "GlycamAngleTemplate") -> dict[str, Any]:
    return {
        "atoms": [_glycam_atom_token_to_jsonable(token) for token in template.atoms],
        "theta0_rad": template.theta0_rad,
        "k_kj_per_mol_rad2": template.k_kj_per_mol_rad2,
    }


def _glycam_torsion_to_jsonable(template: "GlycamTorsionTemplate") -> dict[str, Any]:
    return {
        "atoms": [_glycam_atom_token_to_jsonable(token) for token in template.atoms],
        "periodicity": template.periodicity,
        "phase_rad": template.phase_rad,
        "k_kj_per_mol": template.k_kj_per_mol,
    }


def _glycam_bond_from_jsonable(data: dict[str, Any]) -> "GlycamBondTemplate":
    from poly_csp.forcefield.glycam import GlycamBondTemplate

    return GlycamBondTemplate(
        atoms=tuple(
            _glycam_atom_token_from_jsonable(token)
            for token in list(data["atoms"])
        ),  # type: ignore[arg-type]
        length_nm=float(data["length_nm"]),
        k_kj_per_mol_nm2=float(data["k_kj_per_mol_nm2"]),
    )


def _glycam_angle_from_jsonable(data: dict[str, Any]) -> "GlycamAngleTemplate":
    from poly_csp.forcefield.glycam import GlycamAngleTemplate

    return GlycamAngleTemplate(
        atoms=tuple(
            _glycam_atom_token_from_jsonable(token)
            for token in list(data["atoms"])
        ),  # type: ignore[arg-type]
        theta0_rad=float(data["theta0_rad"]),
        k_kj_per_mol_rad2=float(data["k_kj_per_mol_rad2"]),
    )


def _glycam_torsion_from_jsonable(data: dict[str, Any]) -> "GlycamTorsionTemplate":
    from poly_csp.forcefield.glycam import GlycamTorsionTemplate

    return GlycamTorsionTemplate(
        atoms=tuple(
            _glycam_atom_token_from_jsonable(token)
            for token in list(data["atoms"])
        ),  # type: ignore[arg-type]
        periodicity=int(data["periodicity"]),
        phase_rad=float(data["phase_rad"]),
        k_kj_per_mol=float(data["k_kj_per_mol"]),
    )


def _glycam_params_to_jsonable(params: "GlycamParams") -> dict[str, Any]:
    return {
        "polymer": str(params.polymer),
        "representation": str(params.representation),
        "end_mode": str(params.end_mode),
        "atom_params": [
            {
                "residue_role": str(residue_role),
                "atom_name": str(atom_name),
                "charge_e": atom_params.charge_e,
                "sigma_nm": atom_params.sigma_nm,
                "epsilon_kj_per_mol": atom_params.epsilon_kj_per_mol,
                "residue_name": atom_params.residue_name,
                "source_atom_name": atom_params.source_atom_name,
            }
            for (residue_role, atom_name), atom_params in sorted(params.atom_params.items())
        ],
        "residue_templates": [
            {
                "residue_role": str(residue_role),
                "residue_name": template.residue_name,
                "atom_names": list(template.atom_names),
                "bonds": [_glycam_bond_to_jsonable(bond) for bond in template.bonds],
                "angles": [_glycam_angle_to_jsonable(angle) for angle in template.angles],
                "torsions": [
                    _glycam_torsion_to_jsonable(torsion) for torsion in template.torsions
                ],
            }
            for residue_role, template in sorted(params.residue_templates.items())
        ],
        "linkage_templates": [
            {
                "residue_roles": list(residue_roles),
                "bonds": [_glycam_bond_to_jsonable(bond) for bond in template.bonds],
                "angles": [_glycam_angle_to_jsonable(angle) for angle in template.angles],
                "torsions": [
                    _glycam_torsion_to_jsonable(torsion) for torsion in template.torsions
                ],
            }
            for residue_roles, template in sorted(params.linkage_templates.items())
        ],
        "supported_states": [list(state) for state in params.supported_states],
        "provenance": params.provenance,
    }


def _glycam_params_from_jsonable(data: dict[str, Any]) -> "GlycamParams":
    from poly_csp.forcefield.glycam import (
        GlycamAtomParams,
        GlycamLinkageTemplate,
        GlycamParams,
        GlycamResidueTemplate,
    )

    residue_templates = {
        str(entry["residue_role"]): GlycamResidueTemplate(
            residue_role=str(entry["residue_role"]),  # type: ignore[arg-type]
            residue_name=str(entry["residue_name"]),
            atom_names=tuple(str(name) for name in entry["atom_names"]),
            bonds=tuple(
                _glycam_bond_from_jsonable(bond)
                for bond in list(entry["bonds"])
            ),
            angles=tuple(
                _glycam_angle_from_jsonable(angle)
                for angle in list(entry["angles"])
            ),
            torsions=tuple(
                _glycam_torsion_from_jsonable(torsion)
                for torsion in list(entry["torsions"])
            ),
        )
        for entry in list(data["residue_templates"])
    }
    linkage_templates = {
        tuple(str(role) for role in entry["residue_roles"]): GlycamLinkageTemplate(
            residue_roles=tuple(
                str(role) for role in entry["residue_roles"]
            ),  # type: ignore[arg-type]
            bonds=tuple(
                _glycam_bond_from_jsonable(bond)
                for bond in list(entry["bonds"])
            ),
            angles=tuple(
                _glycam_angle_from_jsonable(angle)
                for angle in list(entry["angles"])
            ),
            torsions=tuple(
                _glycam_torsion_from_jsonable(torsion)
                for torsion in list(entry["torsions"])
            ),
        )
        for entry in list(data["linkage_templates"])
    }
    return GlycamParams(
        polymer=str(data["polymer"]),  # type: ignore[arg-type]
        representation=str(data["representation"]),  # type: ignore[arg-type]
        end_mode=str(data["end_mode"]),  # type: ignore[arg-type]
        atom_params={
            (str(entry["residue_role"]), str(entry["atom_name"])): GlycamAtomParams(
                charge_e=float(entry["charge_e"]),
                sigma_nm=float(entry["sigma_nm"]),
                epsilon_kj_per_mol=float(entry["epsilon_kj_per_mol"]),
                residue_name=str(entry["residue_name"]),
                source_atom_name=str(entry["source_atom_name"]),
            )
            for entry in list(data["atom_params"])
        },
        residue_templates=residue_templates,
        linkage_templates=linkage_templates,
        supported_states=tuple(
            tuple(str(token) for token in state)
            for state in list(data["supported_states"])
        ),
        provenance=dict(data["provenance"]),
    )


def _cache_payload_path(entry_dir: Path) -> Path:
    return entry_dir / "payload.json"


def _write_cache_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_cached_selector_params(entry_dir: Path) -> SelectorFragmentParams | None:
    payload_path = _cache_payload_path(entry_dir)
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PAYLOAD_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("payload_kind") != "selector_fragment":
        return None
    return _selector_params_from_jsonable(dict(payload["payload"]))


def store_cached_selector_params(
    entry_dir: Path,
    *,
    identity: dict[str, Any],
    params: SelectorFragmentParams,
) -> None:
    _write_cache_file(
        _cache_payload_path(entry_dir),
        {
            "schema_version": PAYLOAD_CACHE_SCHEMA_VERSION,
            "payload_kind": "selector_fragment",
            "identity": identity,
            "payload": _selector_params_to_jsonable(params),
        },
    )


def load_cached_connector_params(entry_dir: Path) -> ConnectorParams | None:
    payload_path = _cache_payload_path(entry_dir)
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PAYLOAD_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("payload_kind") != "connector_fragment":
        return None
    return _connector_params_from_jsonable(dict(payload["payload"]))


def store_cached_connector_params(
    entry_dir: Path,
    *,
    identity: dict[str, Any],
    params: ConnectorParams,
) -> None:
    _write_cache_file(
        _cache_payload_path(entry_dir),
        {
            "schema_version": PAYLOAD_CACHE_SCHEMA_VERSION,
            "payload_kind": "connector_fragment",
            "identity": identity,
            "payload": _connector_params_to_jsonable(params),
        },
    )


def load_cached_glycam_params(entry_dir: Path) -> "GlycamParams" | None:
    payload_path = _cache_payload_path(entry_dir)
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PAYLOAD_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("payload_kind") != "glycam_backbone":
        return None
    return _glycam_params_from_jsonable(dict(payload["payload"]))


def store_cached_glycam_params(
    entry_dir: Path,
    *,
    identity: dict[str, Any],
    params: "GlycamParams",
) -> None:
    _write_cache_file(
        _cache_payload_path(entry_dir),
        {
            "schema_version": PAYLOAD_CACHE_SCHEMA_VERSION,
            "payload_kind": "glycam_backbone",
            "identity": identity,
            "payload": _glycam_params_to_jsonable(params),
        },
    )
