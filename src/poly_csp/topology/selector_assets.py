from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files
from typing import Dict, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, PositiveInt
from rdkit import Chem
from rdkit.Chem import AllChem

from poly_csp.topology.selectors import SelectorTemplate, infer_donor_acceptor_atoms


class RotamerGridAssetSpec(BaseModel):
    dihedral_values_deg: Dict[str, tuple[float, ...]]
    max_candidates: PositiveInt = 128


class SelectorAssetSpec(BaseModel):
    name: str
    full_name: str | None = None
    reference_columns: tuple[str, ...] = ()
    reference_backbones: tuple[str, ...] = ()
    mapped_smiles: str
    attach_atom_map_num: PositiveInt
    attach_dummy_map_num: PositiveInt | None = None
    dihedrals: Dict[str, tuple[int, int, int, int]]
    linkage_type: Literal["carbamate", "ester", "ether"] = "carbamate"
    connector_role_by_map_num: Dict[int, str] = Field(default_factory=dict)
    auto_detect_hbond: bool = True
    embed_seed: int = 3501
    rotamer_grid: RotamerGridAssetSpec | None = None


def _asset_refs():
    root = files("poly_csp.assets.selectors")
    return tuple(sorted((ref for ref in root.iterdir() if ref.name.endswith(".yaml")), key=lambda ref: ref.name))


@lru_cache(maxsize=1)
def _asset_ref_by_name() -> dict[str, object]:
    out: dict[str, object] = {}
    for ref in _asset_refs():
        with as_file(ref) as asset_path:
            cfg = OmegaConf.load(asset_path)
        payload = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Selector asset {ref.name!r} did not resolve to a mapping.")
        spec = SelectorAssetSpec.model_validate(payload)
        key = spec.name.strip().lower()
        if key in out:
            raise ValueError(f"Duplicate selector asset name {spec.name!r}.")
        out[key] = ref
    return out


def available_selector_asset_names() -> tuple[str, ...]:
    return tuple(sorted(_asset_ref_by_name().keys()))


def _idx_from_mapnum(mol: Chem.Mol, map_num: int, *, label: str) -> int:
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == int(map_num):
            return int(atom.GetIdx())
    raise ValueError(f"Map number {map_num} not found in selector asset {label!r}.")


def _embed_if_needed(mol: Chem.Mol, *, seed: int, label: str) -> Chem.Mol:
    if mol.GetNumConformers() > 0:
        return mol

    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(with_h, params)
    if status != 0:
        status = AllChem.EmbedMolecule(with_h, useRandomCoords=True, randomSeed=int(seed))
    if status != 0:
        raise RuntimeError(f"RDKit failed to embed selector asset {label!r}.")
    if all(atom.GetAtomicNum() > 0 for atom in with_h.GetAtoms()):
        AllChem.UFFOptimizeMolecule(with_h, maxIters=250)
    return with_h


@lru_cache(maxsize=None)
def load_selector_asset_spec(name: str) -> SelectorAssetSpec:
    key = name.strip().lower()
    ref = _asset_ref_by_name().get(key)
    if ref is None:
        available = ", ".join(available_selector_asset_names())
        raise KeyError(f"Unknown selector asset {name!r}. Available: [{available}]")
    with as_file(ref) as asset_path:
        cfg = OmegaConf.load(asset_path)
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Selector asset {name!r} did not resolve to a mapping.")
    return SelectorAssetSpec.model_validate(payload)


@lru_cache(maxsize=None)
def load_selector_asset_template(name: str) -> SelectorTemplate:
    spec = load_selector_asset_spec(name)
    mol = Chem.MolFromSmiles(spec.mapped_smiles)
    if mol is None:
        raise ValueError(f"Could not parse mapped SMILES for selector asset {spec.name!r}.")
    Chem.SanitizeMol(mol)
    mol = _embed_if_needed(mol, seed=int(spec.embed_seed), label=spec.name)

    attach_dummy_idx = (
        _idx_from_mapnum(mol, int(spec.attach_dummy_map_num), label=spec.name)
        if spec.attach_dummy_map_num is not None
        else None
    )
    attach_atom_idx = _idx_from_mapnum(mol, int(spec.attach_atom_map_num), label=spec.name)
    dihedrals = {
        dihedral_name: tuple(
            _idx_from_mapnum(mol, int(map_num), label=spec.name) for map_num in map_nums
        )
        for dihedral_name, map_nums in spec.dihedrals.items()
    }
    connector_local_roles = {
        _idx_from_mapnum(mol, int(map_num), label=spec.name): role
        for map_num, role in spec.connector_role_by_map_num.items()
    }

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    donors, acceptors = (
        infer_donor_acceptor_atoms(mol) if spec.auto_detect_hbond else ((), ())
    )
    return SelectorTemplate(
        name=spec.name,
        full_name=spec.full_name,
        mol=mol,
        attach_atom_idx=attach_atom_idx,
        attach_dummy_idx=attach_dummy_idx,
        dihedrals=dihedrals,
        donors=tuple(donors),
        acceptors=tuple(acceptors),
        linkage_type=spec.linkage_type,
        connector_local_roles=connector_local_roles,
        features={"donors": tuple(donors), "acceptors": tuple(acceptors)},
        reference_columns=tuple(spec.reference_columns),
        reference_backbones=tuple(spec.reference_backbones),
        rotamer_grid=(
            {}
            if spec.rotamer_grid is None
            else {
                str(name): tuple(float(value) for value in values)
                for name, values in spec.rotamer_grid.dihedral_values_deg.items()
            }
        ),
        rotamer_max_candidates=(
            128 if spec.rotamer_grid is None else int(spec.rotamer_grid.max_candidates)
        ),
    )


def iter_selector_asset_templates() -> tuple[SelectorTemplate, ...]:
    return tuple(load_selector_asset_template(name) for name in available_selector_asset_names())
