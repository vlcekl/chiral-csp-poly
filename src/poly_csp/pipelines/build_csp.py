# src/poly_csp/pipelines/build_csp.py
"""
Minimal working CSP build pipeline.

- Uses Hydra configs from /conf at repo root.
- Builds a helically symmetric amylose/cellulose backbone (coords only -> RDKit mol).
- Optionally attaches selector(s), runs deterministic pre-ordering, and restrained MM.
- Writes PDB + JSON build report (+ optional AMBER scaffold export).

Run (from repo root):
  python -m poly_csp.pipelines.build_csp
  python -m poly_csp.pipelines.build_csp topology.backbone.dp=24
  python -m poly_csp.pipelines.build_csp structure/helix=cellulose_i topology.selector.enabled=true
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
from rdkit import Chem
from omegaconf import DictConfig, OmegaConf

from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.forcefield.runtime_params import load_runtime_params
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.terminals import apply_terminal_mode
from poly_csp.config.schema import (
    BackboneSpec,
    HelixSpec,
    MonomerRepresentation,
    PolymerKind,
    SelectorPoseSpec,
    Site,
)
from poly_csp.forcefield.amber_export import export_amber_artifacts
from poly_csp.forcefield.system_builder import create_system
from poly_csp.io.pdb import write_pdb_from_rdkit
from poly_csp.io.rdkit_io import write_sdf
from poly_csp.structure.pbc import compute_helical_box_vectors, set_box_vectors
from poly_csp.ordering.hbonds import compute_hbond_metrics
from poly_csp.ordering.scoring import (
    bonded_exclusion_pairs,
    min_distance_by_class,
    min_interatomic_distance,
    screw_symmetry_rmsd_from_mol,
    selector_torsion_stats,
)
from poly_csp.ordering.optimize import OrderingSpec, optimize_selector_ordering
from poly_csp.ordering.multi_opt import MultiOptSpec, run_multi_start_optimization

# Optional selector imports (keep pipeline runnable even before selector is implemented).
try:
    from poly_csp.topology.selectors import SelectorRegistry, SelectorTemplate
    from poly_csp.topology.reactions import attach_selector
    from poly_csp.structure.alignment import apply_selector_pose_dihedrals
except (ImportError, ModuleNotFoundError):
    SelectorRegistry = None  # type: ignore
    SelectorTemplate = None  # type: ignore
    apply_selector_pose_dihedrals = None  # type: ignore
    attach_selector = None  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Unexpected error while importing selector modules.") from exc

# Optional MM imports (keep baseline path runnable if OpenMM is unavailable).
try:
    from poly_csp.forcefield.relaxation import RelaxSpec, run_staged_relaxation
except (ImportError, ModuleNotFoundError):
    RelaxSpec = None  # type: ignore
    run_staged_relaxation = None  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Unexpected error while importing OpenMM modules.") from exc


@dataclass(frozen=True)
class QcSpec:
    enabled: bool = True
    exclude_13: bool = True
    exclude_14: bool = False
    min_heavy_distance_A: float = 1.05
    min_backbone_backbone_distance_A: float = 1.05
    min_backbone_selector_distance_A: float = 1.0
    min_selector_selector_distance_A: float = 1.0
    max_screw_symmetry_rmsd_A: float = 0.50
    min_hbond_like_fraction: float = 0.0
    min_hbond_geometric_fraction: float = 0.0
    max_selector_torsion_std_deg: Optional[float] = None
    fail_on_thresholds: bool = False


@dataclass
class BuildReport:
    polymer: str
    dp: int
    monomer_representation: str
    end_mode: str
    end_caps: dict[str, str]
    helix_name: str
    theta_rad: float
    rise_A: float
    residues_per_turn: float
    pitch_A: float
    selector_enabled: bool
    selector_name: Optional[str]
    selector_sites: List[str]
    selector_pose_dihedral_targets_deg: dict[str, float]
    ordering_enabled: bool
    ordering_summary: dict[str, object]
    forcefield_enabled: bool
    forcefield_mode: str
    forcefield_summary: dict[str, object]
    relax_enabled: bool
    relax_mode: str
    relax_summary: dict[str, object]
    qc_min_heavy_distance_A: float
    qc_class_min_distance_A: dict[str, Optional[float]]
    qc_screw_symmetry_rmsd_A: float
    qc_hbond_like_fraction: float
    qc_hbond_geometric_fraction: float
    qc_hbond_like_satisfied_pairs: int
    qc_hbond_geometric_satisfied_pairs: int
    qc_hbond_total_pairs: int
    qc_selector_torsion_stats_deg: dict[str, dict[str, float]]
    qc_thresholds: dict[str, object]
    qc_pass: bool
    qc_fail_reasons: List[str]
    amber_enabled: bool
    amber_summary: dict[str, object]
    output_export_formats: List[str]
    all_atom_atom_count: Optional[int] = None
    all_atom_backbone_h_count: Optional[int] = None
    all_atom_manifest_schema_version: Optional[int] = None
    multi_opt_enabled: bool = False
    multi_opt_rank: int = 0
    multi_opt_total_starts: int = 0
    multi_opt_seed_used: Optional[int] = None


def _cfg_to_helixspec(cfg: DictConfig) -> HelixSpec:
    helix_cfg = cfg.structure.helix
    return HelixSpec(
        name=str(helix_cfg.name),
        theta_rad=float(helix_cfg.theta_rad),
        rise_A=float(helix_cfg.rise_A),
        repeat_residues=int(helix_cfg.repeat_residues)
        if "repeat_residues" in helix_cfg
        else None,
        repeat_turns=int(helix_cfg.repeat_turns)
        if "repeat_turns" in helix_cfg
        else None,
        residues_per_turn=float(helix_cfg.residues_per_turn),
        pitch_A=float(helix_cfg.pitch_A),
        handedness=str(helix_cfg.handedness) if "handedness" in helix_cfg else "right",
    )


def _cfg_to_ordering_spec(cfg: DictConfig) -> OrderingSpec:
    if "ordering" not in cfg or cfg.ordering is None:
        return OrderingSpec()
    payload = OmegaConf.to_container(cfg.ordering, resolve=True)
    if not isinstance(payload, dict):
        return OrderingSpec()
    return OrderingSpec(**payload)


def _cfg_to_multi_opt_spec(cfg: DictConfig) -> MultiOptSpec:
    if "multi_opt" not in cfg or cfg.multi_opt is None:
        return MultiOptSpec()
    payload = OmegaConf.to_container(cfg.multi_opt, resolve=True)
    if not isinstance(payload, dict):
        return MultiOptSpec()
    return MultiOptSpec(**payload)


def _cfg_to_relax_spec(cfg: DictConfig):
    if RelaxSpec is None:
        return None
    relax_cfg = (
        cfg.forcefield.options
        if "forcefield" in cfg
        and cfg.forcefield is not None
        and "options" in cfg.forcefield
        and cfg.forcefield.options is not None
        else {}
    )
    anneal_cfg = (
        relax_cfg.anneal
        if hasattr(relax_cfg, "anneal") and relax_cfg.anneal is not None
        else {}
    )
    enabled = bool(relax_cfg.enabled if "enabled" in relax_cfg else False)
    relax_enabled = bool(
        relax_cfg.relax_enabled if "relax_enabled" in relax_cfg else False
    )
    return RelaxSpec(
        enabled=bool(enabled and relax_enabled),
        positional_k=float(
            relax_cfg.positional_k if "positional_k" in relax_cfg else 5000.0
        ),
        dihedral_k=float(relax_cfg.dihedral_k if "dihedral_k" in relax_cfg else 500.0),
        hbond_k=float(relax_cfg.hbond_k if "hbond_k" in relax_cfg else 50.0),
        n_stages=int(relax_cfg.n_stages if "n_stages" in relax_cfg else 3),
        max_iterations=int(
            relax_cfg.max_iterations if "max_iterations" in relax_cfg else 200
        ),
        freeze_backbone=bool(
            relax_cfg.freeze_backbone if "freeze_backbone" in relax_cfg else True
        ),
        anneal_enabled=bool(anneal_cfg.enabled if "enabled" in anneal_cfg else False),
        t_start_K=float(anneal_cfg.t_start_K if "t_start_K" in anneal_cfg else 50.0),
        t_end_K=float(anneal_cfg.t_end_K if "t_end_K" in anneal_cfg else 350.0),
        anneal_steps=int(anneal_cfg.n_steps if "n_steps" in anneal_cfg else 2000),
        anneal_cool_down=bool(
            anneal_cfg.cool_down if "cool_down" in anneal_cfg else True
        ),
    )


def _forcefield_enabled(cfg: DictConfig) -> bool:
    return bool(
        "forcefield" in cfg
        and cfg.forcefield is not None
        and "options" in cfg.forcefield
        and cfg.forcefield.options is not None
        and "enabled" in cfg.forcefield.options
        and cfg.forcefield.options.enabled
    )


def _runtime_cache_settings(cfg: DictConfig) -> tuple[bool, str | None]:
    options = (
        cfg.forcefield.options
        if "forcefield" in cfg
        and cfg.forcefield is not None
        and "options" in cfg.forcefield
        and cfg.forcefield.options is not None
        else {}
    )
    cache_enabled = bool(options.cache_enabled if "cache_enabled" in options else True)
    cache_dir = (
        str(options.cache_dir)
        if "cache_dir" in options and options.cache_dir is not None
        else None
    )
    return cache_enabled, cache_dir


def _cfg_to_qc_spec(cfg: DictConfig) -> QcSpec:
    qc_cfg = cfg.qc if "qc" in cfg and cfg.qc is not None else {}
    return QcSpec(
        enabled=bool(qc_cfg.enabled if "enabled" in qc_cfg else True),
        exclude_13=bool(qc_cfg.exclude_13 if "exclude_13" in qc_cfg else True),
        exclude_14=bool(qc_cfg.exclude_14 if "exclude_14" in qc_cfg else False),
        min_heavy_distance_A=float(
            qc_cfg.min_heavy_distance_A if "min_heavy_distance_A" in qc_cfg else 1.05
        ),
        min_backbone_backbone_distance_A=float(
            qc_cfg.min_backbone_backbone_distance_A
            if "min_backbone_backbone_distance_A" in qc_cfg
            else 1.05
        ),
        min_backbone_selector_distance_A=float(
            qc_cfg.min_backbone_selector_distance_A
            if "min_backbone_selector_distance_A" in qc_cfg
            else 1.0
        ),
        min_selector_selector_distance_A=float(
            qc_cfg.min_selector_selector_distance_A
            if "min_selector_selector_distance_A" in qc_cfg
            else 1.0
        ),
        max_screw_symmetry_rmsd_A=float(
            qc_cfg.max_screw_symmetry_rmsd_A
            if "max_screw_symmetry_rmsd_A" in qc_cfg
            else 0.50
        ),
        min_hbond_like_fraction=float(
            qc_cfg.min_hbond_like_fraction
            if "min_hbond_like_fraction" in qc_cfg
            else 0.0
        ),
        min_hbond_geometric_fraction=float(
            qc_cfg.min_hbond_geometric_fraction
            if "min_hbond_geometric_fraction" in qc_cfg
            else 0.0
        ),
        max_selector_torsion_std_deg=(
            float(qc_cfg.max_selector_torsion_std_deg)
            if "max_selector_torsion_std_deg" in qc_cfg
            and qc_cfg.max_selector_torsion_std_deg is not None
            else None
        ),
        fail_on_thresholds=bool(
            qc_cfg.fail_on_thresholds if "fail_on_thresholds" in qc_cfg else False
        ),
    )


def _selector_enabled(cfg: DictConfig) -> bool:
    return bool(
        "topology" in cfg
        and cfg.topology is not None
        and "selector" in cfg.topology
        and cfg.topology.selector is not None
        and "enabled" in cfg.topology.selector
        and cfg.topology.selector.enabled
    )


def _ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _heavy_atom_mask_from_rdkit(mol) -> np.ndarray:
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=== poly_csp build_csp ===")
    print(OmegaConf.to_yaml(cfg))

    backbone_cfg = cfg.topology.backbone
    selector_cfg = cfg.topology.selector

    polymer_kind: PolymerKind = str(backbone_cfg.kind)  # type: ignore[assignment]
    dp: int = int(backbone_cfg.dp)
    monomer_representation: MonomerRepresentation = str(
        backbone_cfg.monomer_representation
        if "monomer_representation" in backbone_cfg
        else "anhydro"
    )  # type: ignore[assignment]
    end_mode = str(backbone_cfg.end_mode if "end_mode" in backbone_cfg else "open")
    end_caps = (
        dict(OmegaConf.to_container(backbone_cfg.end_caps, resolve=True))
        if "end_caps" in backbone_cfg and backbone_cfg.end_caps is not None
        else {}
    )

    output_export_formats = (
        [str(x).strip().lower() for x in cfg.output.export_formats]
        if "output" in cfg and "export_formats" in cfg.output
        else ["pdb"]
    )
    amber_cfg_enabled = bool(
        "amber" in cfg and cfg.amber is not None and "enabled" in cfg.amber and cfg.amber.enabled
    )
    if not amber_cfg_enabled:
        output_export_formats = [fmt for fmt in output_export_formats if fmt != "amber"]
    if amber_cfg_enabled and "amber" not in output_export_formats:
        output_export_formats.append("amber")

    amber_dir = (
        str(cfg.amber.dir)
        if "amber" in cfg and cfg.amber is not None and "dir" in cfg.amber
        else "amber"
    )
    amber_charge_model = (
        str(cfg.amber.charge_model)
        if "amber" in cfg and cfg.amber is not None and "charge_model" in cfg.amber
        else "bcc"
    )
    amber_net_charge = (
        cfg.amber.net_charge
        if "amber" in cfg and cfg.amber is not None and "net_charge" in cfg.amber
        else "auto"
    )

    unsupported_formats = [
        fmt for fmt in output_export_formats if fmt not in {"pdb", "amber", "sdf"}
    ]
    if unsupported_formats:
        raise NotImplementedError(
            "Requested export formats are not implemented yet: "
            + ", ".join(sorted(set(unsupported_formats)))
        )

    helix = _cfg_to_helixspec(cfg)
    backbone = BackboneSpec(
        polymer=polymer_kind,
        dp=dp,
        monomer_representation=monomer_representation,
        end_mode=end_mode,  # type: ignore[arg-type]
        end_caps=end_caps,
        helix=helix,
    )
    ordering_spec = _cfg_to_ordering_spec(cfg)
    qc_spec = _cfg_to_qc_spec(cfg)

    forcefield_enabled = _forcefield_enabled(cfg)
    forcefield_mode = "runtime" if forcefield_enabled else "none"
    relax_spec = _cfg_to_relax_spec(cfg)
    relax_requested = bool(relax_spec is not None and relax_spec.enabled)
    if relax_requested and (run_staged_relaxation is None or relax_spec is None):
        raise RuntimeError(
            "Relaxation requested but OpenMM modules are unavailable in this environment."
        )

    outdir = _ensure_outdir(
        cfg.output.dir if "output" in cfg and "dir" in cfg.output else "outputs"
    )

    # ---- Stage 1/2: topology state -> direct explicit-H backbone build.
    template = make_glucose_template(
        backbone.polymer,
        monomer_representation=backbone.monomer_representation,
    )
    topology_mol = polymerize(
        template=template,
        dp=backbone.dp,
        linkage="1-4",
        anomer="alpha" if backbone.polymer == "amylose" else "beta",
    )
    topology_mol = apply_terminal_mode(
        mol=topology_mol,
        mode=backbone.end_mode,
        caps=backbone.end_caps,
        representation=backbone.monomer_representation,
    )
    mol_poly = build_backbone_structure(topology_mol, helix_spec=helix).mol

    # ---- Stage 2b: compute and store PBC box vectors for periodic mode.
    is_periodic = str(backbone.end_mode) == "periodic"
    if is_periodic:
        Lx, Ly, Lz = compute_helical_box_vectors(
            mol=mol_poly, helix=helix, dp=backbone.dp, padding_A=30.0,
        )
        set_box_vectors(mol_poly, Lx, Ly, Lz)
        print(f"  PBC box: {Lx:.1f} x {Ly:.1f} x {Lz:.1f} Å")

    # ---- Stage 3: optional selector attachment and deterministic pose setup.
    selector_enabled = _selector_enabled(cfg)
    selector_name: Optional[str] = None
    selector_sites: List[Site] = []
    selector_pose = SelectorPoseSpec()
    selector: SelectorTemplate | None = None
    ordering_summary: dict[str, object] = {}
    ordering_applied = False
    ranked_results = None

    if selector_enabled:
        if (
            SelectorRegistry is None
            or attach_selector is None
            or apply_selector_pose_dihedrals is None
        ):
            raise RuntimeError(
                "Selector attachment requested but selector modules are not available."
            )

        selector_name = str(selector_cfg.name)
        selector_sites = [str(s) for s in selector_cfg.sites]  # type: ignore[assignment]
        selector = SelectorRegistry.get(selector_name)

        if "pose" in selector_cfg and selector_cfg.pose is not None:
            pose_payload = OmegaConf.to_container(selector_cfg.pose, resolve=True)
            if isinstance(pose_payload, dict):
                selector_pose = SelectorPoseSpec(**pose_payload)

        for res_i in range(backbone.dp):
            for site in selector_sites:
                mol_poly = attach_selector(
                    mol_polymer=mol_poly,
                    residue_index=res_i,
                    site=site,  # type: ignore[arg-type]
                    selector=selector,
                    mode="bond_from_OH_oxygen",
                )
                if selector_pose.dihedral_targets_deg:
                    mol_poly = apply_selector_pose_dihedrals(
                        mol=mol_poly,
                        residue_index=res_i,
                        site=site,  # type: ignore[arg-type]
                        pose_spec=selector_pose,
                        selector=selector,
                    )

        if ordering_spec.enabled:
            multi_opt_spec = _cfg_to_multi_opt_spec(cfg)
            if multi_opt_spec.enabled:
                # Multi-start mode: run N optimizations, keep top K.
                ranked_results = run_multi_start_optimization(
                    mol=mol_poly,
                    selector=selector,
                    sites=selector_sites,
                    dp=backbone.dp,
                    ordering_spec=ordering_spec,
                    multi_spec=multi_opt_spec,
                )
                # Use best result as the primary mol.
                best = ranked_results[0]
                mol_poly = best.mol
                ordering_summary = best.summary
            else:
                mol_poly, ordering_summary = optimize_selector_ordering(
                    mol=mol_poly,
                    selector=selector,
                    sites=selector_sites,
                    dp=backbone.dp,
                    spec=ordering_spec,
                )
                ranked_results = None
            ordering_applied = True
        else:
            ranked_results = None

    amber_summary: dict[str, object] = {"enabled": False}
    forcefield_summary: dict[str, object] = {"enabled": False, "mode": "none"}
    runtime_params = None
    runtime_mol = build_forcefield_molecule(mol_poly).mol
    runtime_cache_enabled, runtime_cache_dir = _runtime_cache_settings(cfg)

    # ---- Stage 4a: optional runtime forcefield build.
    if forcefield_enabled:
        runtime_params = load_runtime_params(
            runtime_mol,
            selector_template=selector,
            work_dir=outdir / "runtime_params",
            cache_enabled=runtime_cache_enabled,
            cache_dir=runtime_cache_dir,
        )
        built_system = create_system(
            runtime_mol,
            glycam_params=runtime_params.glycam,
            selector_params_by_name=runtime_params.selector_params_by_name,
            connector_params_by_key=runtime_params.connector_params_by_key,
            nonbonded_mode="full",
            mixing_rules_cfg=(
                OmegaConf.to_container(cfg.forcefield, resolve=True)
                if "forcefield" in cfg and cfg.forcefield is not None
                else None
            ),
        )
        forcefield_summary = {
            "enabled": True,
            "mode": forcefield_mode,
            "parameter_backend": "runtime_component_merge",
            "nonbonded_mode": built_system.nonbonded_mode,
            "particle_count": int(built_system.system.getNumParticles()),
            "force_count": int(built_system.system.getNumForces()),
            "topology_manifest_size": len(built_system.topology_manifest),
            "component_counts": dict(built_system.component_counts),
            "exception_summary": dict(built_system.exception_summary),
            "source_manifest": dict(built_system.source_manifest),
            "runtime_param_cache": asdict(runtime_params.cache_summary),
        }

    # ---- Stage 5: optional restrained relaxation.
    relax_summary: dict[str, object] = {"enabled": False}
    relax_enabled = False
    if relax_spec is not None and bool(relax_spec.enabled):
        if run_staged_relaxation is None:
            raise RuntimeError(
                "Relaxation requested but run_staged_relaxation is unavailable."
            )
        runtime_mol, relax_summary = run_staged_relaxation(
            mol=runtime_mol,
            spec=relax_spec,
            selector=selector,
            runtime_params=runtime_params,
            mixing_rules_cfg=(
                OmegaConf.to_container(cfg.forcefield, resolve=True)
                if "forcefield" in cfg and cfg.forcefield is not None
                else None
            ),
        )
        relax_enabled = True
        mol_poly = Chem.Mol(runtime_mol)

    # ---- Stage 6: QC metrics and threshold evaluation.
    conf = mol_poly.GetConformer(0)
    xyz = np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))
    heavy_mask = _heavy_atom_mask_from_rdkit(mol_poly)

    max_path_length = 1 + int(qc_spec.exclude_13) + int(qc_spec.exclude_14)
    excluded_pairs = (
        bonded_exclusion_pairs(mol_poly, max_path_length=max_path_length)
        if qc_spec.enabled
        else set()
    )
    qc_min_dist = float(min_interatomic_distance(xyz, heavy_mask, excluded_pairs))
    qc_class_dist_raw = min_distance_by_class(mol_poly, xyz, heavy_mask, excluded_pairs)
    qc_class_dist = {k: _finite_or_none(v) for k, v in qc_class_dist_raw.items()}

    k = int(helix.repeat_residues) if helix.repeat_residues else 1
    qc_sym_rmsd = float(screw_symmetry_rmsd_from_mol(mol_poly, helix=helix, k=k))

    qc_hbond_like_fraction = 0.0
    qc_hbond_geometric_fraction = 0.0
    qc_hbond_like_satisfied_pairs = 0
    qc_hbond_geometric_satisfied_pairs = 0
    qc_hbond_total_pairs = 0
    qc_selector_torsions: dict[str, dict[str, float]] = {}
    if selector is not None:
        hb = compute_hbond_metrics(
            mol=mol_poly,
            selector=selector,
            max_distance_A=ordering_spec.max_distance_A,
            neighbor_window=ordering_spec.neighbor_window,
            min_donor_angle_deg=ordering_spec.min_donor_angle_deg,
            min_acceptor_angle_deg=ordering_spec.min_acceptor_angle_deg,
        )
        qc_hbond_like_fraction = float(hb.like_fraction)
        qc_hbond_geometric_fraction = float(hb.geometric_fraction)
        qc_hbond_like_satisfied_pairs = int(hb.like_satisfied_pairs)
        qc_hbond_geometric_satisfied_pairs = int(hb.geometric_satisfied_pairs)
        qc_hbond_total_pairs = int(hb.total_pairs)
        qc_selector_torsions = selector_torsion_stats(
            mol=mol_poly,
            selector_dihedrals=selector.dihedrals,
            attach_dummy_idx=selector.attach_dummy_idx,
        )

    qc_thresholds = {
        "min_heavy_distance_A": float(qc_spec.min_heavy_distance_A),
        "min_backbone_backbone_distance_A": float(
            qc_spec.min_backbone_backbone_distance_A
        ),
        "min_backbone_selector_distance_A": float(
            qc_spec.min_backbone_selector_distance_A
        ),
        "min_selector_selector_distance_A": float(
            qc_spec.min_selector_selector_distance_A
        ),
        "max_screw_symmetry_rmsd_A": float(qc_spec.max_screw_symmetry_rmsd_A),
        "min_hbond_like_fraction": float(qc_spec.min_hbond_like_fraction),
        "min_hbond_geometric_fraction": float(qc_spec.min_hbond_geometric_fraction),
        "max_selector_torsion_std_deg": qc_spec.max_selector_torsion_std_deg,
        "exclude_13": bool(qc_spec.exclude_13),
        "exclude_14": bool(qc_spec.exclude_14),
    }

    qc_fail_reasons: List[str] = []
    if qc_spec.enabled:
        if qc_min_dist < qc_spec.min_heavy_distance_A:
            qc_fail_reasons.append(
                f"min_heavy_distance_A={qc_min_dist:.3f} < {qc_spec.min_heavy_distance_A:.3f}"
            )
        bb = qc_class_dist_raw["backbone_backbone"]
        if np.isfinite(bb) and bb < qc_spec.min_backbone_backbone_distance_A:
            qc_fail_reasons.append(
                "backbone_backbone distance "
                f"{bb:.3f} < {qc_spec.min_backbone_backbone_distance_A:.3f}"
            )
        bs = qc_class_dist_raw["backbone_selector"]
        if np.isfinite(bs) and bs < qc_spec.min_backbone_selector_distance_A:
            qc_fail_reasons.append(
                "backbone_selector distance "
                f"{bs:.3f} < {qc_spec.min_backbone_selector_distance_A:.3f}"
            )
        ss = qc_class_dist_raw["selector_selector"]
        if np.isfinite(ss) and ss < qc_spec.min_selector_selector_distance_A:
            qc_fail_reasons.append(
                "selector_selector distance "
                f"{ss:.3f} < {qc_spec.min_selector_selector_distance_A:.3f}"
            )
        if qc_sym_rmsd > qc_spec.max_screw_symmetry_rmsd_A:
            qc_fail_reasons.append(
                f"screw_symmetry_rmsd_A={qc_sym_rmsd:.3f} > {qc_spec.max_screw_symmetry_rmsd_A:.3f}"
            )
        if (
            selector is not None
            and qc_hbond_like_fraction < qc_spec.min_hbond_like_fraction
        ):
            qc_fail_reasons.append(
                "hbond_like_fraction="
                f"{qc_hbond_like_fraction:.3f} < {qc_spec.min_hbond_like_fraction:.3f}"
            )
        if (
            selector is not None
            and qc_hbond_geometric_fraction < qc_spec.min_hbond_geometric_fraction
        ):
            qc_fail_reasons.append(
                "hbond_geometric_fraction="
                f"{qc_hbond_geometric_fraction:.3f} < "
                f"{qc_spec.min_hbond_geometric_fraction:.3f}"
            )
        if qc_spec.max_selector_torsion_std_deg is not None:
            for torsion_name, stats in qc_selector_torsions.items():
                std_deg = float(stats.get("std_deg", 0.0))
                if std_deg > qc_spec.max_selector_torsion_std_deg:
                    qc_fail_reasons.append(
                        f"{torsion_name}.std_deg={std_deg:.3f} > "
                        f"{qc_spec.max_selector_torsion_std_deg:.3f}"
                    )

    qc_pass = len(qc_fail_reasons) == 0

    # ---- Stage 8: optional AMBER export from the final coordinates.
    amber_enabled = "amber" in output_export_formats
    if amber_enabled:
        from poly_csp.structure.pbc import get_box_vectors_A as _get_bv
        export_mol = runtime_mol if forcefield_enabled else mol_poly
        _bv = _get_bv(export_mol) if is_periodic else None
        amber_summary = export_amber_artifacts(
            mol=export_mol,
            outdir=outdir / amber_dir,
            model_name="model",
            charge_model=amber_charge_model,
            net_charge=amber_net_charge,
            polymer=polymer_kind,
            dp=dp,
            selector_mol=(
                selector.mol if selector is not None else None
            ),
            periodic=is_periodic,
            box_vectors_A=_bv,
        )

    forcefield_result = build_forcefield_molecule(runtime_mol if forcefield_enabled else mol_poly)
    final_mol = forcefield_result.mol
    all_atom_stats = {
        "all_atom_atom_count": int(final_mol.GetNumAtoms()),
        "all_atom_backbone_h_count": sum(
            1
            for entry in forcefield_result.manifest
            if entry.component == "backbone" and entry.atom_index != entry.parent_heavy_index
        ),
        "all_atom_manifest_schema_version": (
            int(final_mol.GetIntProp("_poly_csp_manifest_schema_version"))
            if final_mol.HasProp("_poly_csp_manifest_schema_version")
            else None
        ),
    }
    report = BuildReport(
        polymer=str(backbone.polymer),
        dp=int(backbone.dp),
        monomer_representation=str(backbone.monomer_representation),
        end_mode=str(backbone.end_mode),
        end_caps=dict(backbone.end_caps),
        helix_name=str(helix.name),
        theta_rad=float(helix.theta_rad),
        rise_A=float(helix.rise_A),
        residues_per_turn=float(helix.residues_per_turn),
        pitch_A=float(helix.pitch_A),
        selector_enabled=bool(selector_enabled),
        selector_name=selector_name,
        selector_sites=[str(s) for s in selector_sites],
        selector_pose_dihedral_targets_deg=dict(selector_pose.dihedral_targets_deg),
        ordering_enabled=bool(ordering_applied),
        ordering_summary=ordering_summary,
        forcefield_enabled=bool(forcefield_enabled),
        forcefield_mode=str(forcefield_mode),
        forcefield_summary=forcefield_summary,
        relax_enabled=bool(relax_enabled),
        relax_mode="two_stage_runtime" if relax_enabled else "none",
        relax_summary=relax_summary,
        qc_min_heavy_distance_A=qc_min_dist,
        qc_class_min_distance_A=qc_class_dist,
        qc_screw_symmetry_rmsd_A=qc_sym_rmsd,
        qc_hbond_like_fraction=qc_hbond_like_fraction,
        qc_hbond_geometric_fraction=qc_hbond_geometric_fraction,
        qc_hbond_like_satisfied_pairs=qc_hbond_like_satisfied_pairs,
        qc_hbond_geometric_satisfied_pairs=qc_hbond_geometric_satisfied_pairs,
        qc_hbond_total_pairs=qc_hbond_total_pairs,
        qc_selector_torsion_stats_deg=qc_selector_torsions,
        qc_thresholds=qc_thresholds,
        qc_pass=bool(qc_pass),
        qc_fail_reasons=qc_fail_reasons,
        amber_enabled=bool(amber_enabled),
        amber_summary=amber_summary,
        output_export_formats=output_export_formats,
        all_atom_atom_count=all_atom_stats["all_atom_atom_count"],
        all_atom_backbone_h_count=all_atom_stats["all_atom_backbone_h_count"],
        all_atom_manifest_schema_version=all_atom_stats["all_atom_manifest_schema_version"],
        multi_opt_enabled=bool(ranked_results is not None and len(ranked_results) > 0),
        multi_opt_rank=1 if ranked_results else 0,
        multi_opt_total_starts=len(ranked_results) if ranked_results else 0,
        multi_opt_seed_used=int(ranked_results[0].seed_used) if ranked_results else None,
    )

    # ---- Write outputs.
    pdb_path = outdir / "model.pdb"
    json_path = outdir / "build_report.json"
    cfg_path = outdir / "resolved_config.yaml"

    if "pdb" in output_export_formats:
        write_pdb_from_rdkit(final_mol, pdb_path)

    sdf_path = outdir / "model.sdf"
    if "sdf" in output_export_formats:
        write_sdf(final_mol, sdf_path)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2)

    with open(cfg_path, "w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(cfg))

    wrote_lines = []
    if "pdb" in output_export_formats:
        wrote_lines.append(f"  {pdb_path}")
    if "sdf" in output_export_formats:
        wrote_lines.append(f"  {sdf_path}")
    wrote_lines.extend([f"  {json_path}", f"  {cfg_path}"])
    print("\nWrote:\n" + "\n".join(wrote_lines))
    if amber_enabled and "files" in amber_summary:
        print(f"  {amber_summary['manifest']}")
    print("\nQC:")
    print(f"  pass:                         {qc_pass}")
    print(f"  min heavy-atom distance (A):  {qc_min_dist:.3f}")
    print(f"  screw symmetry RMSD (A):      {qc_sym_rmsd:.3f}")
    print(f"  hbond-like fraction:          {qc_hbond_like_fraction:.3f}")
    print(f"  hbond-geometric fraction:     {qc_hbond_geometric_fraction:.3f}")
    if qc_fail_reasons:
        print("  failures:")
        for reason in qc_fail_reasons:
            print(f"    - {reason}")

    if qc_spec.fail_on_thresholds and not qc_pass:
        raise RuntimeError("QC thresholds failed; see build_report.json for details.")

    # ---- Write ranked results for multi-start optimization.
    if ranked_results and len(ranked_results) > 1:
        ranking_entries = []
        for result in ranked_results:
            rank_dir = _ensure_outdir(outdir / f"ranked_{result.rank:03d}")
            rank_final_mol = build_forcefield_molecule(result.mol).mol
            if "pdb" in output_export_formats:
                write_pdb_from_rdkit(rank_final_mol, rank_dir / "model.pdb")
            if "sdf" in output_export_formats:
                write_sdf(rank_final_mol, rank_dir / "model.sdf")
            rank_report = {
                "rank": result.rank,
                "score": result.score,
                "seed_used": result.seed_used,
                "ordering_summary": result.summary,
            }
            with open(rank_dir / "build_report.json", "w", encoding="utf-8") as h:
                json.dump(rank_report, h, indent=2)
            ranking_entries.append({
                "rank": result.rank,
                "score": result.score,
                "seed_used": result.seed_used,
                "dir": str(rank_dir),
            })
        ranking_path = outdir / "ranking_summary.json"
        with open(ranking_path, "w", encoding="utf-8") as h:
            json.dump({"n_starts": len(ranked_results), "ranking": ranking_entries}, h, indent=2)
        print(f"  {ranking_path}")
        for entry in ranking_entries:
            print(f"    rank {entry['rank']}: score={entry['score']:.4f}  seed={entry['seed_used']}")


if __name__ == "__main__":
    main()
