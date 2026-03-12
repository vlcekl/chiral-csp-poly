#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from poly_csp.cache_versions import (
    RUNTIME_PAYLOAD_CACHE_SCHEMA_VERSION,
    RUNTIME_PAYLOAD_MODEL_VERSION,
)
from poly_csp.forcefield.connectors import load_connector_params
from poly_csp.forcefield.gaff import load_selector_fragment_params
from poly_csp.forcefield.payload_cache import (
    _connector_params_to_jsonable,
    _selector_params_to_jsonable,
    connector_cache_identity,
    selector_cache_identity,
)
from poly_csp.topology.selector_assets import (
    available_selector_asset_names,
    load_selector_asset_template,
)

POLYMERS = ("amylose", "cellulose")
SITES = ("C2", "C3", "C6")
MONOMER_REPRESENTATION = "natural_oh"
ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "poly_csp"
    / "assets"
    / "runtime_params"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _selector_entry(
    selector_name: str, work_root: Path
) -> tuple[str, dict[str, object]]:
    selector_template = load_selector_asset_template(selector_name)
    selector_key, selector_identity = selector_cache_identity(selector_template)
    selector_params = load_selector_fragment_params(
        selector_template=selector_template,
        work_dir=work_root / "selector" / selector_name,
    )
    return (
        selector_key,
        {
            "identity": selector_identity,
            "payload": _selector_params_to_jsonable(
                replace(
                    selector_params,
                    source_prmtop=f"seeded://selector/{selector_name}/{selector_key}",
                )
            ),
        },
    )


def _connector_entry(
    selector_name: str,
    polymer: str,
    site: str,
    work_root: Path,
) -> tuple[str, dict[str, object]]:
    selector_template = load_selector_asset_template(selector_name)
    connector_key, connector_identity = connector_cache_identity(
        polymer,
        selector_template,
        site,
        monomer_representation=MONOMER_REPRESENTATION,
    )
    connector_params = load_connector_params(
        polymer=polymer,
        selector_template=selector_template,
        site=site,
        monomer_representation=MONOMER_REPRESENTATION,
        work_dir=work_root / "connector" / selector_name / polymer / site.lower(),
    )
    return (
        connector_key,
        {
            "identity": connector_identity,
            "payload": _connector_params_to_jsonable(
                replace(
                    connector_params,
                    source_prmtop=(
                        f"seeded://connector/{polymer}/{selector_name}/{site.lower()}/{connector_key}"
                    ),
                )
            ),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker count for selector/connector parameterization.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selector_names = tuple(available_selector_asset_names())
    selector_entries: dict[str, dict[str, object]] = {}
    connector_entries: dict[str, dict[str, object]] = {}

    with TemporaryDirectory(prefix="polycsp_seed_runtime_") as tmp_root:
        work_root = Path(tmp_root)
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            selector_futures = {
                pool.submit(_selector_entry, selector_name, work_root): selector_name
                for selector_name in selector_names
            }
            for future in as_completed(selector_futures):
                selector_name = selector_futures[future]
                key, entry = future.result()
                selector_entries[key] = entry
                print(f"[selector] {selector_name}", flush=True)

            connector_futures = {
                pool.submit(
                    _connector_entry, selector_name, polymer, site, work_root
                ): (
                    selector_name,
                    polymer,
                    site,
                )
                for selector_name in selector_names
                for polymer in POLYMERS
                for site in SITES
            }
            for future in as_completed(connector_futures):
                selector_name, polymer, site = connector_futures[future]
                key, entry = future.result()
                connector_entries[key] = entry
                print(f"[connector] {selector_name} {polymer} {site}", flush=True)

    _write_json(
        ASSET_ROOT / "selectors.json",
        {
            "schema_version": RUNTIME_PAYLOAD_CACHE_SCHEMA_VERSION,
            "model_version": RUNTIME_PAYLOAD_MODEL_VERSION,
            "payload_kind": "selector_fragment_seed_catalog",
            "selectors": list(selector_names),
            "entries": selector_entries,
        },
    )
    _write_json(
        ASSET_ROOT / "connectors.json",
        {
            "schema_version": RUNTIME_PAYLOAD_CACHE_SCHEMA_VERSION,
            "model_version": RUNTIME_PAYLOAD_MODEL_VERSION,
            "payload_kind": "connector_fragment_seed_catalog",
            "selectors": list(selector_names),
            "polymers": list(POLYMERS),
            "sites": list(SITES),
            "entries": connector_entries,
        },
    )


if __name__ == "__main__":
    main()
