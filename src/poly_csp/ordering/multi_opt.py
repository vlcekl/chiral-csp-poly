"""Multi-start selector ordering optimization.

Runs N independent ordering optimizations with different random seeds,
optionally in parallel, and returns the top-K ranked by score.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import PropertyPickleOptions

from poly_csp.chemistry.selectors import SelectorTemplate
from poly_csp.config.schema import Site
from poly_csp.ordering.rotamers import RotamerGridSpec
from poly_csp.ordering.symmetry_opt import OrderingSpec, optimize_selector_ordering


@dataclass(frozen=True)
class MultiOptSpec:
    """Configuration for multi-start optimization."""

    enabled: bool = False
    n_starts: int = 5
    top_k: int = 3
    seed: Optional[int] = None  # master seed; None = random
    strategy: str = "multi_start"  # future: "basin_hopping"
    n_workers: int = 0  # 0 = auto (os.cpu_count()), 1 = serial


@dataclass
class RankedResult:
    """A single optimization result with its ranking metadata."""

    rank: int
    score: float
    mol: Chem.Mol
    summary: Dict[str, object]
    seed_used: int


def _run_single_start(
    mol_binary: bytes,
    selector: SelectorTemplate,
    sites: Sequence[str],
    dp: int,
    ordering_spec: OrderingSpec,
    grid: Optional[RotamerGridSpec],
    child_seed: int,
) -> Tuple[float, bytes, Dict[str, object], int]:
    """Run a single optimization start — designed to be called in a worker process."""
    mol = Chem.Mol(mol_binary)
    opt_mol, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=sites,
        dp=dp,
        spec=ordering_spec,
        grid=grid,
        seed=child_seed,
    )
    score = float(summary.get("final_score", float("-inf")))
    return score, opt_mol.ToBinary(PropertyPickleOptions.AllProps), summary, child_seed


def run_multi_start_optimization(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    sites: Iterable[Site],
    dp: int,
    ordering_spec: OrderingSpec,
    multi_spec: MultiOptSpec,
    grid: RotamerGridSpec | None = None,
) -> List[RankedResult]:
    """Run *n_starts* independent ordering optimizations and return *top_k* best.

    Each start uses a different child seed derived from *multi_spec.seed*
    (or system entropy if None), producing different rotamer grid traversal
    orders and therefore potentially different local minima.

    When *n_workers* > 1 (or 0 for auto), runs are executed in parallel
    using a process pool for near-linear speedup on multi-core machines.

    Returns a list of :class:`RankedResult` sorted by descending score
    (rank 1 = best).
    """
    if multi_spec.strategy != "multi_start":
        raise ValueError(
            f"Unsupported multi-opt strategy {multi_spec.strategy!r}. "
            "Currently supported: 'multi_start'."
        )

    n_starts = max(1, int(multi_spec.n_starts))
    top_k = max(1, min(int(multi_spec.top_k), n_starts))
    sites_list = [str(s) for s in sites]

    # Derive reproducible child seeds from a master SeedSequence.
    master = np.random.SeedSequence(multi_spec.seed)
    child_seeds = [int(cs.generate_state(1)[0]) for cs in master.spawn(n_starts)]

    # Resolve worker count.
    n_workers = int(multi_spec.n_workers)
    if n_workers <= 0:
        n_workers = min(n_starts, os.cpu_count() or 1)
    n_workers = min(n_workers, n_starts)

    # Serialize mol once — workers deserialize from binary.
    mol_binary = mol.ToBinary(PropertyPickleOptions.AllProps)

    raw_results: List[Tuple[float, bytes, Dict[str, object], int]]

    if n_workers == 1:
        # Serial path (also used in tests / debugging).
        raw_results = [
            _run_single_start(
                mol_binary, selector, sites_list, dp,
                ordering_spec, grid, seed,
            )
            for seed in child_seeds
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    _run_single_start,
                    mol_binary, selector, sites_list, dp,
                    ordering_spec, grid, seed,
                )
                for seed in child_seeds
            ]
            raw_results = [f.result() for f in futures]

    # Sort descending by score.
    raw_results.sort(key=lambda r: r[0], reverse=True)

    ranked: List[RankedResult] = []
    for i, (score, opt_mol_binary, summary, seed_used) in enumerate(raw_results[:top_k]):
        ranked.append(
            RankedResult(
                rank=i + 1,
                score=score,
                mol=Chem.Mol(opt_mol_binary),
                summary=summary,
                seed_used=seed_used,
            )
        )

    return ranked
