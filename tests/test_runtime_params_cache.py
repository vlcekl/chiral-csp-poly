from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.connectors import ConnectorAtomParams, ConnectorParams
from poly_csp.forcefield.gaff import SelectorAtomParams, SelectorFragmentParams
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.payload_cache import (
    PAYLOAD_CACHE_SCHEMA_VERSION,
    connector_cache_dir,
    load_cached_connector_params,
    selector_cache_dir,
)
from poly_csp.forcefield.runtime_params import load_runtime_params
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.selectors import SelectorRegistry
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.terminals import apply_terminal_mode


def _helix() -> HelixSpec:
    return HelixSpec(
        name="cache_test_helix",
        theta_rad=-4.71238898038469,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=4.933333333333334,
        handedness="left",
    )


def _forcefield_selector_mol(site: str):
    selector = SelectorRegistry.get("35dmpc")
    template = make_glucose_template("amylose", monomer_representation="anhydro")
    topology = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    topology = apply_terminal_mode(
        mol=topology,
        mode="open",
        caps={},
        representation="anhydro",
    )
    structure = build_backbone_structure(topology, _helix()).mol
    structure = attach_selector(
        mol_polymer=structure,
        residue_index=0,
        site=site,
        selector=selector,
    )
    return build_forcefield_molecule(structure).mol, selector


def _selector_payload(selector_name: str, work_dir: str | Path | None) -> SelectorFragmentParams:
    work_path = None if work_dir is None else Path(work_dir)
    return SelectorFragmentParams(
        selector_name=selector_name,
        atom_params={
            "S000": SelectorAtomParams(
                atom_name="S000",
                charge_e=-0.12,
                sigma_nm=0.33,
                epsilon_kj_per_mol=0.21,
            )
        },
        bonds=(),
        angles=(),
        torsions=(),
        source_prmtop=None if work_path is None else str(work_path / "selector.prmtop"),
        fragment_atom_count=1,
    )


def _connector_payload(
    selector_name: str,
    site: str,
    work_dir: str | Path | None,
) -> ConnectorParams:
    work_path = None if work_dir is None else Path(work_dir)
    return ConnectorParams(
        polymer="amylose",
        selector_name=selector_name,
        site=site,
        monomer_representation="natural_oh",
        linkage_type="ether",
        atom_params={
            "SL_000": ConnectorAtomParams(
                atom_name="SL_000",
                charge_e=0.07,
                sigma_nm=0.29,
                epsilon_kj_per_mol=0.09,
            )
        },
        connector_role_atom_names={},
        bonds=(),
        angles=(),
        torsions=(),
        source_prmtop=None if work_path is None else str(work_path / f"connector_{site}.prmtop"),
        fragment_atom_count=1,
    )


def test_load_runtime_params_reuses_selector_and_connector_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mol, selector = _forcefield_selector_mol("C6")
    cache_dir = tmp_path / "runtime_cache"
    calls = {"selector": 0, "connector": 0}
    glycam = SimpleNamespace(kind="glycam")

    monkeypatch.setattr(
        "poly_csp.forcefield.runtime_params.load_glycam_params",
        lambda **kwargs: glycam,
    )

    def fake_selector_loader(selector_template, charge_model="bcc", net_charge=0, work_dir=None):
        calls["selector"] += 1
        assert charge_model == "bcc"
        assert net_charge == 0
        return _selector_payload(selector_template.name, work_dir)

    def fake_connector_loader(
        polymer,
        selector_template,
        site,
        charge_model="bcc",
        net_charge=0,
        monomer_representation="natural_oh",
        work_dir=None,
    ):
        calls["connector"] += 1
        assert polymer == "amylose"
        assert charge_model == "bcc"
        assert net_charge == 0
        assert monomer_representation == "natural_oh"
        return _connector_payload(selector_template.name, site, work_dir)

    monkeypatch.setattr(
        "poly_csp.forcefield.runtime_params.load_selector_fragment_params",
        fake_selector_loader,
    )
    monkeypatch.setattr(
        "poly_csp.forcefield.runtime_params.load_connector_params",
        fake_connector_loader,
    )

    first = load_runtime_params(
        mol,
        selector_template=selector,
        cache_dir=cache_dir,
    )
    second = load_runtime_params(
        mol,
        selector_template=selector,
        cache_dir=cache_dir,
    )

    assert calls == {"selector": 1, "connector": 1}
    assert first.glycam is glycam
    assert second.glycam is glycam
    assert first.selector_params_by_name == second.selector_params_by_name
    assert first.connector_params_by_key == second.connector_params_by_key
    assert first.cache_summary.selector_hits == 0
    assert first.cache_summary.selector_misses == 1
    assert first.cache_summary.connector_hits == 0
    assert first.cache_summary.connector_misses == 1
    assert second.cache_summary.selector_hits == 1
    assert second.cache_summary.selector_misses == 0
    assert second.cache_summary.connector_hits == 1
    assert second.cache_summary.connector_misses == 0
    assert first.source_manifest["selector"][selector.name]["cache"]["hit"] is False
    assert first.source_manifest["connector"][f"{selector.name}:C6"]["cache"]["hit"] is False
    assert second.source_manifest["selector"][selector.name]["cache"]["hit"] is True
    assert second.source_manifest["connector"][f"{selector.name}:C6"]["cache"]["hit"] is True

    selector_entry, _ = selector_cache_dir(cache_dir, selector)
    connector_entry, _ = connector_cache_dir(
        cache_dir,
        polymer="amylose",
        selector_template=selector,
        site="C6",
        monomer_representation="natural_oh",
    )
    assert (selector_entry / "payload.json").exists()
    assert (connector_entry / "payload.json").exists()


def test_load_runtime_params_invalidates_connector_cache_by_site(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mol_c6, selector_c6 = _forcefield_selector_mol("C6")
    mol_c2, selector_c2 = _forcefield_selector_mol("C2")
    cache_dir = tmp_path / "runtime_cache"
    calls = {"selector": 0, "connector": 0}

    monkeypatch.setattr(
        "poly_csp.forcefield.runtime_params.load_glycam_params",
        lambda **kwargs: SimpleNamespace(kind="glycam"),
    )

    def fake_selector_loader(selector_template, charge_model="bcc", net_charge=0, work_dir=None):
        calls["selector"] += 1
        return _selector_payload(selector_template.name, work_dir)

    def fake_connector_loader(
        polymer,
        selector_template,
        site,
        charge_model="bcc",
        net_charge=0,
        monomer_representation="natural_oh",
        work_dir=None,
    ):
        calls["connector"] += 1
        return _connector_payload(selector_template.name, site, work_dir)

    monkeypatch.setattr(
        "poly_csp.forcefield.runtime_params.load_selector_fragment_params",
        fake_selector_loader,
    )
    monkeypatch.setattr(
        "poly_csp.forcefield.runtime_params.load_connector_params",
        fake_connector_loader,
    )

    first = load_runtime_params(
        mol_c6,
        selector_template=selector_c6,
        cache_dir=cache_dir,
    )
    second = load_runtime_params(
        mol_c2,
        selector_template=selector_c2,
        cache_dir=cache_dir,
    )

    assert calls == {"selector": 1, "connector": 2}
    assert first.cache_summary.selector_hits == 0
    assert first.cache_summary.selector_misses == 1
    assert first.cache_summary.connector_hits == 0
    assert first.cache_summary.connector_misses == 1
    assert second.cache_summary.selector_hits == 1
    assert second.cache_summary.selector_misses == 0
    assert second.cache_summary.connector_hits == 0
    assert second.cache_summary.connector_misses == 1
    assert second.source_manifest["selector"][selector_c2.name]["cache"]["hit"] is True
    assert second.source_manifest["connector"][f"{selector_c2.name}:C2"]["cache"]["hit"] is False


def test_connector_cache_dir_is_polymer_specific(tmp_path: Path) -> None:
    selector = SelectorRegistry.get("35dmpc")
    amylose_dir, _ = connector_cache_dir(
        tmp_path,
        polymer="amylose",
        selector_template=selector,
        site="C6",
        monomer_representation="natural_oh",
    )
    cellulose_dir, _ = connector_cache_dir(
        tmp_path,
        polymer="cellulose",
        selector_template=selector,
        site="C6",
        monomer_representation="natural_oh",
    )

    assert amylose_dir != cellulose_dir


def test_load_cached_connector_params_rejects_stale_schema(tmp_path: Path) -> None:
    entry_dir = tmp_path / "connector_entry"
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "payload.json").write_text(
        json.dumps(
            {
                "schema_version": PAYLOAD_CACHE_SCHEMA_VERSION - 1,
                "payload_kind": "connector_fragment",
                "identity": {"kind": "connector_fragment"},
                "payload": {},
            }
        ),
        encoding="utf-8",
    )

    assert load_cached_connector_params(entry_dir) is None
