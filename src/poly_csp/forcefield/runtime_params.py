"""Canonical runtime parameter loading for the supported forcefield slice."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from poly_csp.forcefield.connectors import ConnectorParams, load_connector_params
from poly_csp.forcefield.gaff import SelectorFragmentParams, load_selector_fragment_params
from poly_csp.forcefield.glycam import GlycamParams, load_glycam_params
from poly_csp.forcefield.payload_cache import (
    connector_cache_dir,
    load_cached_connector_params,
    load_cached_selector_params,
    resolve_runtime_cache_dir,
    selector_cache_dir,
    store_cached_connector_params,
    store_cached_selector_params,
)


@dataclass(frozen=True)
class RuntimeParamCacheSummary:
    enabled: bool
    cache_dir: str | None
    glycam_hits: int = 0
    glycam_misses: int = 0
    selector_hits: int = 0
    selector_misses: int = 0
    connector_hits: int = 0
    connector_misses: int = 0


@dataclass(frozen=True)
class RuntimeParams:
    glycam: GlycamParams
    selector_params_by_name: dict[str, SelectorFragmentParams]
    connector_params_by_key: dict[tuple[str, str], ConnectorParams]
    cache_summary: RuntimeParamCacheSummary
    source_manifest: dict[str, object] = field(default_factory=dict)


def _cache_meta(
    *,
    cache_enabled: bool,
    cache_hit: bool,
    cache_entry_dir: Path | None,
) -> dict[str, object]:
    return {
        "cache": {
            "enabled": bool(cache_enabled),
            "hit": bool(cache_hit) if cache_enabled else False,
            "kind": (
                "hit"
                if cache_enabled and cache_hit
                else ("build" if cache_enabled else "disabled")
            ),
            "entry_dir": None if cache_entry_dir is None else str(cache_entry_dir),
        }
    }


def _load_or_build_selector_params(
    *,
    selector_template,
    work_dir: Path | None,
    cache_enabled: bool,
    cache_dir: str | Path | None,
) -> tuple[SelectorFragmentParams, dict[str, object]]:
    if not cache_enabled:
        return (
            load_selector_fragment_params(
                selector_template=selector_template,
                work_dir=None if work_dir is None else work_dir / "selector",
            ),
            {
                "selector_name": str(selector_template.name),
                **_cache_meta(
                    cache_enabled=False,
                    cache_hit=False,
                    cache_entry_dir=None if work_dir is None else work_dir / "selector",
                ),
            },
        )

    entry_dir, identity = selector_cache_dir(cache_dir, selector_template)
    cached = load_cached_selector_params(entry_dir)
    if cached is not None:
        return (
            cached,
            {
                "selector_name": str(selector_template.name),
                **_cache_meta(
                    cache_enabled=True,
                    cache_hit=True,
                    cache_entry_dir=entry_dir,
                ),
            },
        )

    params = load_selector_fragment_params(
        selector_template=selector_template,
        work_dir=entry_dir,
    )
    store_cached_selector_params(entry_dir, identity=identity, params=params)
    return (
        params,
        {
            "selector_name": str(selector_template.name),
            **_cache_meta(
                cache_enabled=True,
                cache_hit=False,
                cache_entry_dir=entry_dir,
            ),
        },
    )


def _load_or_build_connector_params(
    *,
    polymer: str,
    selector_template,
    site: str,
    work_dir: Path | None,
    cache_enabled: bool,
    cache_dir: str | Path | None,
) -> tuple[ConnectorParams, dict[str, object]]:
    if not cache_enabled:
        return (
            load_connector_params(
                polymer=polymer,  # type: ignore[arg-type]
                selector_template=selector_template,
                site=site,  # type: ignore[arg-type]
                monomer_representation="natural_oh",
                work_dir=None if work_dir is None else work_dir / f"connector_{site.lower()}",
            ),
            {
                "selector_name": str(selector_template.name),
                "site": str(site),
                **_cache_meta(
                    cache_enabled=False,
                    cache_hit=False,
                    cache_entry_dir=(
                        None if work_dir is None else work_dir / f"connector_{site.lower()}"
                    ),
                ),
            },
        )

    entry_dir, identity = connector_cache_dir(
        cache_dir,
        polymer=polymer,  # type: ignore[arg-type]
        selector_template=selector_template,
        site=site,  # type: ignore[arg-type]
        monomer_representation="natural_oh",
    )
    cached = load_cached_connector_params(entry_dir)
    if cached is not None:
        return (
            cached,
            {
                "selector_name": str(selector_template.name),
                "site": str(site),
                **_cache_meta(
                    cache_enabled=True,
                    cache_hit=True,
                    cache_entry_dir=entry_dir,
                ),
            },
        )

    params = load_connector_params(
        polymer=polymer,  # type: ignore[arg-type]
        selector_template=selector_template,
        site=site,  # type: ignore[arg-type]
        monomer_representation="natural_oh",
        work_dir=entry_dir,
    )
    store_cached_connector_params(entry_dir, identity=identity, params=params)
    return (
        params,
        {
            "selector_name": str(selector_template.name),
            "site": str(site),
            **_cache_meta(
                cache_enabled=True,
                cache_hit=False,
                cache_entry_dir=entry_dir,
            ),
        },
    )


def load_runtime_params(
    mol: Chem.Mol,
    selector_template=None,
    work_dir: Path | None = None,
    *,
    cache_enabled: bool = True,
    cache_dir: str | Path | None = None,
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
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
    )
    resolved_cache_dir = (
        str(resolve_runtime_cache_dir(cache_dir)) if cache_enabled else None
    )
    glycam_provenance = (
        glycam.provenance if isinstance(getattr(glycam, "provenance", None), dict) else {}
    )
    glycam_cache_meta = (
        glycam_provenance.get("cache", {})
        if isinstance(glycam_provenance.get("cache", {}), dict)
        else {}
    )
    glycam_hit_count = (
        1
        if cache_enabled and bool(glycam_cache_meta.get("hit"))
        else 0
    )
    glycam_miss_count = (
        1
        if cache_enabled and not bool(glycam_cache_meta.get("hit"))
        else 0
    )

    selector_params_by_name: dict[str, SelectorFragmentParams] = {}
    connector_params_by_key: dict[tuple[str, str], ConnectorParams] = {}
    source_manifest: dict[str, object] = {"glycam": dict(glycam_provenance)}
    selector_hit_count = 0
    selector_miss_count = 0
    connector_hit_count = 0
    connector_miss_count = 0

    selector_instance_atoms = [
        atom for atom in mol.GetAtoms() if atom.HasProp("_poly_csp_selector_instance")
    ]
    if not selector_instance_atoms:
        return RuntimeParams(
            glycam=glycam,
            selector_params_by_name=selector_params_by_name,
            connector_params_by_key=connector_params_by_key,
            cache_summary=RuntimeParamCacheSummary(
                enabled=bool(cache_enabled),
                cache_dir=resolved_cache_dir,
                glycam_hits=glycam_hit_count,
                glycam_misses=glycam_miss_count,
            ),
            source_manifest=source_manifest,
        )

    if selector_template is None:
        raise ValueError(
            "Selector-bearing runtime parameter loading requires the SelectorTemplate."
        )

    selector_params, selector_meta = _load_or_build_selector_params(
        selector_template=selector_template,
        work_dir=work_dir,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
    )
    selector_params_by_name[selector_template.name] = selector_params
    source_manifest.setdefault("selector", {})[selector_template.name] = selector_meta
    if bool(selector_meta["cache"]["enabled"]) and bool(selector_meta["cache"]["hit"]):
        selector_hit_count += 1
    elif bool(selector_meta["cache"]["enabled"]):
        selector_miss_count += 1

    sites = sorted(
        {
            atom.GetProp("_poly_csp_site")
            for atom in selector_instance_atoms
            if atom.HasProp("_poly_csp_site")
        }
    )
    for site in sites:
        connector_params, connector_meta = _load_or_build_connector_params(
            polymer=polymer,
            selector_template=selector_template,
            site=site,
            work_dir=work_dir,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )
        connector_params_by_key[(selector_template.name, site)] = connector_params
        source_manifest.setdefault("connector", {})[
            f"{selector_template.name}:{site}"
        ] = connector_meta
        if bool(connector_meta["cache"]["enabled"]) and bool(connector_meta["cache"]["hit"]):
            connector_hit_count += 1
        elif bool(connector_meta["cache"]["enabled"]):
            connector_miss_count += 1

    return RuntimeParams(
        glycam=glycam,
        selector_params_by_name=selector_params_by_name,
        connector_params_by_key=connector_params_by_key,
        cache_summary=RuntimeParamCacheSummary(
            enabled=bool(cache_enabled),
            cache_dir=resolved_cache_dir,
            glycam_hits=glycam_hit_count,
            glycam_misses=glycam_miss_count,
            selector_hits=selector_hit_count,
            selector_misses=selector_miss_count,
            connector_hits=connector_hit_count,
            connector_misses=connector_miss_count,
        ),
        source_manifest=source_manifest,
    )
