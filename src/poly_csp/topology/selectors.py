# poly_csp/topology/selectors.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, TypeAlias

from rdkit import Chem


AnchorDihedralAtomRef: TypeAlias = Literal["site_carbon", "site_oxygen"]
SelectorDihedralAtomRef: TypeAlias = int | AnchorDihedralAtomRef
SelectorDihedralSpec: TypeAlias = Tuple[
    SelectorDihedralAtomRef,
    SelectorDihedralAtomRef,
    SelectorDihedralAtomRef,
    SelectorDihedralAtomRef,
]


def infer_donor_acceptor_atoms(mol: Chem.Mol) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Lightweight donor/acceptor inference for selector plugins.
    """

    def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
        if atom.GetAtomicNum() != 6:
            return False
        return any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(atom).GetAtomicNum() in {8, 16}
            for bond in atom.GetBonds()
        )

    def _is_amide_like_nitrogen(atom: Chem.Atom) -> bool:
        if atom.GetAtomicNum() != 7:
            return False
        return any(
            bond.GetBondType() == Chem.BondType.SINGLE
            and _is_carbonyl_carbon(bond.GetOtherAtom(atom))
            for bond in atom.GetBonds()
        )

    donors: list[int] = []
    acceptors: list[int] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        if z == 7:
            has_h = atom.GetTotalNumHs(includeNeighbors=True) > 0
            if has_h and charge <= 0:
                donors.append(idx)
            if charge <= 0 and not _is_amide_like_nitrogen(atom):
                acceptors.append(idx)
        elif z == 8:
            if charge <= 0:
                acceptors.append(idx)
        elif z == 16 and charge <= 0:
            acceptors.append(idx)
    return tuple(donors), tuple(acceptors)


@dataclass(frozen=True)
class SelectorTemplate:
    name: str
    mol: Chem.Mol
    attach_atom_idx: int  # atom to bond from (typically carbonyl carbon)
    dihedrals: Dict[str, Tuple[int, int, int, int]]  # selector-local indices
    anchor_dihedrals: Dict[str, SelectorDihedralSpec] = field(default_factory=dict)
    donors: Tuple[int, ...] = ()
    acceptors: Tuple[int, ...] = ()
    attach_dummy_idx: int | None = None  # optional [*] replaced at attachment
    linkage_type: Literal["carbamate", "ester", "ether"] = "carbamate"
    connector_local_roles: Dict[int, str] = field(default_factory=dict)
    features: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    full_name: str | None = None
    reference_columns: Tuple[str, ...] = ()
    reference_backbones: Tuple[str, ...] = ()
    rotamer_grid: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    anchor_rotamer_grid: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    rotamer_max_candidates: int = 128


def selector_from_smiles(
    name: str,
    smiles: str,
    attach_atom_idx: int,
    dihedrals: Dict[str, Tuple[int, int, int, int]],
    anchor_dihedrals: Dict[str, SelectorDihedralSpec] | None = None,
    attach_dummy_idx: int | None = None,
    linkage_type: Literal["carbamate", "ester", "ether"] = "carbamate",
    connector_local_roles: Dict[int, str] | None = None,
    auto_detect_hbond: bool = True,
) -> SelectorTemplate:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid selector SMILES for {name!r}.")
    Chem.SanitizeMol(mol)

    donors, acceptors = infer_donor_acceptor_atoms(mol) if auto_detect_hbond else ((), ())
    return SelectorTemplate(
        name=name,
        full_name=name,
        mol=mol,
        attach_atom_idx=int(attach_atom_idx),
        attach_dummy_idx=attach_dummy_idx,
        dihedrals=dict(dihedrals),
        anchor_dihedrals=(
            {}
            if anchor_dihedrals is None
            else {
                str(name): tuple(spec)
                for name, spec in anchor_dihedrals.items()
            }
        ),
        donors=tuple(donors),
        acceptors=tuple(acceptors),
        linkage_type=linkage_type,
        connector_local_roles=(
            {} if connector_local_roles is None else dict(connector_local_roles)
        ),
        features={"donors": tuple(donors), "acceptors": tuple(acceptors)},
    )


class SelectorRegistry:
    _reg: Dict[str, SelectorTemplate] = {}
    _assets_loaded: bool = False

    @classmethod
    def _norm(cls, name: str) -> str:
        return name.strip().lower()

    @classmethod
    def _load_assets(cls) -> None:
        if cls._assets_loaded:
            return
        from poly_csp.topology.selector_assets import iter_selector_asset_templates

        for template in iter_selector_asset_templates():
            cls.register(template)
        cls._assets_loaded = True

    @classmethod
    def register(cls, template: SelectorTemplate) -> None:
        key = cls._norm(template.name)
        existing = cls._reg.get(key)
        if existing is not None and existing is not template:
            raise ValueError(f"Selector {template.name!r} is already registered.")
        cls._reg[key] = template

    @classmethod
    def get(cls, name: str) -> SelectorTemplate:
        cls._load_assets()
        key = cls._norm(name)
        if key not in cls._reg:
            available = ", ".join(sorted(cls._reg.keys()))
            raise KeyError(f"Unknown selector {name!r}. Available: [{available}]")
        return cls._reg[key]

    @classmethod
    def available(cls) -> tuple[str, ...]:
        cls._load_assets()
        return tuple(sorted(cls._reg.keys()))
