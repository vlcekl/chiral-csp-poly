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
from poly_csp.forcefield.export_bundle import prepare_export_bundle
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
    RuntimeForcefieldOptions,
    SelectorPoseSpec,
    Site,
)
from poly_csp.forcefield.amber_export import export_amber_artifacts
from poly_csp.forcefield.system_builder import create_system
from poly_csp.io.pdb import write_pdb_from_rdkit
from poly_csp.io.pdbqt import write_receptor_pdbqt
from poly_csp.io.rdkit_io import write_sdf
from poly_csp.io.vina import VinaBoxSpec, build_vina_box, write_vina_box
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


@dataclass(frozen=True)
class DockingSpec:
    box_buffer_A: float = 6.0
    window_residues: Optional[int] = None


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
    docking_enabled: bool
    docking_summary: dict[str, object]
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


def _cfg_to_forcefield_options(cfg: DictConfig) -> RuntimeForcefieldOptions:
    if (
        "forcefield" not in cfg
        or cfg.forcefield is None
        or "options" not in cfg.forcefield
        or cfg.forcefield.options is None
    ):
        return RuntimeForcefieldOptions(enabled=False)

    payload = OmegaConf.to_container(cfg.forcefield.options, resolve=True)
    if not isinstance(payload, dict):
        return RuntimeForcefieldOptions(enabled=False)
    return RuntimeForcefieldOptions(**payload)


def _cfg_to_relax_spec(options: RuntimeForcefieldOptions):
    if RelaxSpec is None:
        return None
    return RelaxSpec(
        enabled=bool(options.enabled and options.relax_enabled),
        positional_k=float(options.positional_k),
        dihedral_k=float(options.dihedral_k),
        hbond_k=float(options.hbond_k),
        freeze_backbone=bool(options.freeze_backbone),
        soft_n_stages=int(options.soft_n_stages),
        soft_max_iterations=int(options.soft_max_iterations),
        full_max_iterations=int(options.full_max_iterations),
        final_restraint_factor=float(options.final_restraint_factor),
        anneal_enabled=bool(options.anneal.enabled),
        t_start_K=float(options.anneal.t_start_K),
        t_end_K=float(options.anneal.t_end_K),
        anneal_steps=int(options.anneal.n_steps),
        anneal_cool_down=bool(options.anneal.cool_down),
    )


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


def _cfg_to_docking_spec(cfg: DictConfig) -> DockingSpec:
    docking_cfg = cfg.docking if "docking" in cfg and cfg.docking is not None else {}
    return DockingSpec(
        box_buffer_A=float(
            docking_cfg.box_buffer_A if "box_buffer_A" in docking_cfg else 6.0
        ),
        window_residues=(
            int(docking_cfg.window_residues)
            if "window_residues" in docking_cfg
            and docking_cfg.window_residues is not None
            else None
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


def _export_runtime_artifacts(
    *,
    mol: Chem.Mol,
    outdir: Path,
    export_formats: list[str],
    helix: HelixSpec,
    docking_spec: DockingSpec,
    runtime_params,
    built_system=None,
    mixing_rules_cfg: dict | None = None,
    repulsion_k_kj_per_mol_nm2: float = 800.0,
    repulsion_cutoff_nm: float = 0.6,
    model_name: str = "model",
) -> tuple[dict[str, object], dict[str, object]]:
    amber_summary: dict[str, object] = {"enabled": False}
    docking_summary: dict[str, object] = {"enabled": False}
    requested = {fmt for fmt in export_formats if fmt in {"amber", "pdbqt"}}
    if not requested:
        return amber_summary, docking_summary

    bundle = prepare_export_bundle(
        mol,
        runtime_params=runtime_params,
        system_build=built_system,
        mixing_rules_cfg=mixing_rules_cfg,
        repulsion_k_kj_per_mol_nm2=float(repulsion_k_kj_per_mol_nm2),
        repulsion_cutoff_nm=float(repulsion_cutoff_nm),
    )

    if "pdbqt" in requested:
        pdbqt_summary = write_receptor_pdbqt(bundle, outdir / "receptor.pdbqt")
        vina_box = build_vina_box(
            bundle.mol,
            helix=helix,
            spec=VinaBoxSpec(
                buffer_A=float(docking_spec.box_buffer_A),
                window_residues=docking_spec.window_residues,
            ),
        )
        box_summary = write_vina_box(vina_box, outdir / "vina_box.txt")
        docking_summary = {
            "enabled": True,
            "backend": "native_openmm_charge_export",
            "source_manifest": dict(bundle.system_build.source_manifest),
            "receptor": pdbqt_summary,
            "vina_box": box_summary,
        }

    if "amber" in requested:
        amber_summary = export_amber_artifacts(
            bundle,
            outdir=outdir,
            model_name=model_name,
        )

    return amber_summary, docking_summary


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

    unsupported_formats = [
        fmt for fmt in output_export_formats if fmt not in {"pdb", "amber", "sdf", "pdbqt"}
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
    docking_spec = _cfg_to_docking_spec(cfg)
    forcefield_options = _cfg_to_forcefield_options(cfg)

    forcefield_enabled = bool(forcefield_options.enabled)
    forcefield_mode = "runtime" if forcefield_enabled else "none"
    requires_runtime_export = any(
        fmt in {"amber", "pdbqt"} for fmt in output_export_formats
    )
    if requires_runtime_export and not forcefield_enabled:
        raise ValueError(
            "Requested amber/pdbqt export requires forcefield.options.enabled=true "
            "because these artifacts are emitted from the canonical full runtime system."
        )
    relax_spec = _cfg_to_relax_spec(forcefield_options)
    relax_requested = bool(relax_spec is not None and relax_spec.enabled)
    if relax_requested and (run_staged_relaxation is None or relax_spec is None):
        raise RuntimeError(
            "Relaxation requested but OpenMM modules are unavailable in this environment."
        )

    outdir = _ensure_outdir(
        cfg.output.dir if "output" in cfg and "dir" in cfg.output else "outputs"
    )
    mixing_rules_cfg = (
        OmegaConf.to_container(cfg.forcefield, resolve=True)
        if "forcefield" in cfg and cfg.forcefield is not None
        else None
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

    amber_summary: dict[str, object] = {"enabled": False}
    docking_summary: dict[str, object] = {"enabled": False}
    forcefield_summary: dict[str, object] = {"enabled": False, "mode": "none"}
    runtime_mol = build_forcefield_molecule(mol_poly).mol
    runtime_params = None
    built_system = None
    runtime_cache_enabled = bool(forcefield_options.cache_enabled)
    runtime_cache_dir = forcefield_options.cache_dir

    # ---- Stage 3b: optional ordering on the canonical runtime molecule.
    if ordering_spec.enabled:
        if selector is None or not selector_sites:
            ordering_summary = {
                "enabled": False,
                "skipped": True,
                "reason": "no_selector",
            }
        else:
            if not forcefield_enabled:
                raise ValueError(
                    "Selector ordering requires forcefield.options.enabled=true "
                    "because ordering runs on the canonical runtime system."
                )
            runtime_params = load_runtime_params(
                runtime_mol,
                selector_template=selector,
                work_dir=outdir / "runtime_params",
                cache_enabled=runtime_cache_enabled,
                cache_dir=runtime_cache_dir,
            )
            multi_opt_spec = _cfg_to_multi_opt_spec(cfg)
            if multi_opt_spec.enabled:
                ranked_results = run_multi_start_optimization(
                    mol=runtime_mol,
                    selector=selector,
                    sites=selector_sites,
                    dp=backbone.dp,
                    ordering_spec=ordering_spec,
                    multi_spec=multi_opt_spec,
                    runtime_params=runtime_params,
                    work_dir=outdir / "runtime_params",
                    cache_enabled=runtime_cache_enabled,
                    cache_dir=runtime_cache_dir,
                    mixing_rules_cfg=mixing_rules_cfg,
                )
                best = ranked_results[0]
                runtime_mol = best.mol
                ordering_summary = best.summary
            else:
                runtime_mol, ordering_summary = optimize_selector_ordering(
                    mol=runtime_mol,
                    selector=selector,
                    sites=selector_sites,
                    dp=backbone.dp,
                    spec=ordering_spec,
                    runtime_params=runtime_params,
                    work_dir=outdir / "runtime_params",
                    cache_enabled=runtime_cache_enabled,
                    cache_dir=runtime_cache_dir,
                    mixing_rules_cfg=mixing_rules_cfg,
                )
                ranked_results = None
            ordering_applied = True
    else:
        ranked_results = None

    # ---- Stage 4a: optional runtime forcefield build.
    if forcefield_enabled:
        if runtime_params is None:
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
            parameter_provenance=runtime_params.source_manifest,
            nonbonded_mode="full",
            repulsion_k_kj_per_mol_nm2=float(
                forcefield_options.soft_repulsion_k_kj_per_mol_nm2
            ),
            repulsion_cutoff_nm=float(forcefield_options.soft_repulsion_cutoff_nm),
            mixing_rules_cfg=mixing_rules_cfg,
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
            "bonded_term_summary": asdict(built_system.bonded_term_summary),
            "force_inventory": asdict(built_system.force_inventory),
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
            soft_repulsion_k_kj_per_mol_nm2=float(
                forcefield_options.soft_repulsion_k_kj_per_mol_nm2
            ),
            soft_repulsion_cutoff_nm=float(forcefield_options.soft_repulsion_cutoff_nm),
            mixing_rules_cfg=mixing_rules_cfg,
        )
        relax_enabled = True

    # ---- Stage 6: QC metrics and threshold evaluation.
    qc_mol = runtime_mol
    conf = qc_mol.GetConformer(0)
    xyz = np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))
    heavy_mask = _heavy_atom_mask_from_rdkit(qc_mol)

    max_path_length = 1 + int(qc_spec.exclude_13) + int(qc_spec.exclude_14)
    excluded_pairs = (
        bonded_exclusion_pairs(qc_mol, max_path_length=max_path_length)
        if qc_spec.enabled
        else set()
    )
    qc_min_dist = float(min_interatomic_distance(xyz, heavy_mask, excluded_pairs))
    qc_class_dist_raw = min_distance_by_class(qc_mol, xyz, heavy_mask, excluded_pairs)
    qc_class_dist = {k: _finite_or_none(v) for k, v in qc_class_dist_raw.items()}

    k = int(helix.repeat_residues) if helix.repeat_residues else 1
    qc_sym_rmsd = float(screw_symmetry_rmsd_from_mol(qc_mol, helix=helix, k=k))

    qc_hbond_like_fraction = 0.0
    qc_hbond_geometric_fraction = 0.0
    qc_hbond_like_satisfied_pairs = 0
    qc_hbond_geometric_satisfied_pairs = 0
    qc_hbond_total_pairs = 0
    qc_selector_torsions: dict[str, dict[str, float]] = {}
    if selector is not None:
        hb = compute_hbond_metrics(
            mol=qc_mol,
            selector=selector,
            max_distance_A=ordering_spec.hbond_max_distance_A,
            neighbor_window=ordering_spec.hbond_neighbor_window,
            min_donor_angle_deg=ordering_spec.hbond_min_donor_angle_deg,
            min_acceptor_angle_deg=ordering_spec.hbond_min_acceptor_angle_deg,
        )
        qc_hbond_like_fraction = float(hb.like_fraction)
        qc_hbond_geometric_fraction = float(hb.geometric_fraction)
        qc_hbond_like_satisfied_pairs = int(hb.like_satisfied_pairs)
        qc_hbond_geometric_satisfied_pairs = int(hb.geometric_satisfied_pairs)
        qc_hbond_total_pairs = int(hb.total_pairs)
        qc_selector_torsions = selector_torsion_stats(
            mol=qc_mol,
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

    forcefield_result = build_forcefield_molecule(runtime_mol)
    final_mol = forcefield_result.mol
    amber_enabled = "amber" in output_export_formats
    docking_enabled = "pdbqt" in output_export_formats
    amber_summary, docking_summary = _export_runtime_artifacts(
        mol=final_mol,
        outdir=outdir,
        export_formats=output_export_formats,
        helix=helix,
        docking_spec=docking_spec,
        runtime_params=runtime_params,
        built_system=(built_system if not relax_enabled else None),
        mixing_rules_cfg=mixing_rules_cfg,
        repulsion_k_kj_per_mol_nm2=float(
            forcefield_options.soft_repulsion_k_kj_per_mol_nm2
        ),
        repulsion_cutoff_nm=float(forcefield_options.soft_repulsion_cutoff_nm),
        model_name="model",
    )
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
        docking_enabled=bool(docking_enabled),
        docking_summary=docking_summary,
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
    if docking_enabled and "receptor" in docking_summary:
        wrote_lines.append(f"  {docking_summary['receptor']['file']}")
    if docking_enabled and "vina_box" in docking_summary:
        wrote_lines.append(f"  {docking_summary['vina_box']['file']}")
    if amber_enabled and "files" in amber_summary:
        wrote_lines.append(f"  {amber_summary['files']['prmtop']}")
        wrote_lines.append(f"  {amber_summary['files']['inpcrd']}")
    wrote_lines.extend([f"  {json_path}", f"  {cfg_path}"])
    print("\nWrote:\n" + "\n".join(wrote_lines))
    if amber_enabled and "manifest" in amber_summary:
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
            rank_amber_summary, rank_docking_summary = _export_runtime_artifacts(
                mol=rank_final_mol,
                outdir=rank_dir,
                export_formats=output_export_formats,
                helix=helix,
                docking_spec=docking_spec,
                runtime_params=runtime_params,
                built_system=None,
                mixing_rules_cfg=mixing_rules_cfg,
                repulsion_k_kj_per_mol_nm2=float(
                    forcefield_options.soft_repulsion_k_kj_per_mol_nm2
                ),
                repulsion_cutoff_nm=float(forcefield_options.soft_repulsion_cutoff_nm),
                model_name="model",
            )
            rank_report = {
                "rank": result.rank,
                "score": result.score,
                "seed_used": result.seed_used,
                "ordering_summary": result.summary,
                "amber_summary": rank_amber_summary,
                "docking_summary": rank_docking_summary,
            }
            with open(rank_dir / "build_report.json", "w", encoding="utf-8") as h:
                json.dump(rank_report, h, indent=2)
            ranking_entries.append({
                "rank": result.rank,
                "score": result.score,
                "seed_used": result.seed_used,
                "dir": str(rank_dir),
                "amber_enabled": bool(rank_amber_summary.get("enabled", False)),
                "docking_enabled": bool(rank_docking_summary.get("enabled", False)),
            })
        ranking_path = outdir / "ranking_summary.json"
        with open(ranking_path, "w", encoding="utf-8") as h:
            json.dump({"n_starts": len(ranked_results), "ranking": ranking_entries}, h, indent=2)
        print(f"  {ranking_path}")
        for entry in ranking_entries:
            print(f"    rank {entry['rank']}: score={entry['score']:.4f}  seed={entry['seed_used']}")


if __name__ == "__main__":
    main()
