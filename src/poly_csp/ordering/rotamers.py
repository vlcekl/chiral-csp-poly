from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence

from poly_csp.config.schema import SelectorPoseSpec
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class RotamerGridSpec:
    dihedral_values_deg: Dict[str, Sequence[float]]
    max_candidates: int = 128


def _unique_preserve_order(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    seen: set[float] = set()
    for v in values:
        vf = float(v)
        if vf not in seen:
            seen.add(vf)
            out.append(vf)
    return out


def default_rotamer_grid(selector: SelectorTemplate) -> RotamerGridSpec:
    if selector.rotamer_grid:
        return RotamerGridSpec(
            dihedral_values_deg={
                str(name): tuple(float(value) for value in values)
                for name, values in selector.rotamer_grid.items()
            },
            max_candidates=int(selector.rotamer_max_candidates),
        )
    return RotamerGridSpec(
        dihedral_values_deg={"tau_link": (-120.0, 0.0, 120.0)},
        max_candidates=32,
    )


def enumerate_pose_library(grid: RotamerGridSpec) -> List[SelectorPoseSpec]:
    names = [name for name in sorted(grid.dihedral_values_deg.keys())]
    if not names:
        return [SelectorPoseSpec(dihedral_targets_deg={})]

    values_per_name = [_unique_preserve_order(grid.dihedral_values_deg[name]) for name in names]
    combos = list(product(*values_per_name))
    combos = combos[: int(grid.max_candidates)]

    poses: List[SelectorPoseSpec] = []
    for combo in combos:
        dihedrals = {name: float(val) for name, val in zip(names, combo)}
        poses.append(SelectorPoseSpec(dihedral_targets_deg=dihedrals))

    if not poses:
        poses.append(SelectorPoseSpec(dihedral_targets_deg={}))
    return poses
