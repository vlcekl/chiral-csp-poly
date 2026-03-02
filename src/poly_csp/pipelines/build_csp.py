# src/poly_csp/pipelines/build_csp.py
"""
Minimal working CSP build pipeline.

- Uses Hydra configs from /conf at repo root.
- Builds a helically symmetric amylose/cellulose backbone (coords only -> RDKit mol).
- Optionally attaches selector(s), runs deterministic pre-ordering, and restrained MM.
- Writes PDB + JSON build report (+ optional AMBER scaffold export).

Run (from repo root):
  python -m poly_csp.pipelines.build_csp
  python -m poly_csp.pipelines.build_csp polymer.dp=24 helix=admcp_chiralpak_ad selector.enabled=true
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.chemistry.terminals import apply_terminal_mode
from poly_csp.config.schema import (
    BackboneSpec,
    HelixSpec,
    MonomerRepresentation,
    PolymerKind,
    SelectorPoseSpec,
    Site,
)
from poly_csp.io.amber import export_amber_artifacts
from poly_csp.io.pdb import write_pdb_from_rdkit
from poly_csp.io.rdkit_io import write_sdf
from poly_csp.ordering.hbonds import compute_hbond_metrics
from poly_csp.ordering.scoring import (
    bonded_exclusion_pairs,
    min_distance_by_class,
    min_interatomic_distance,
    screw_symmetry_rmsd_from_mol,
    selector_torsion_stats,
)
from poly_csp.ordering.symmetry_opt import OrderingSpec, optimize_selector_ordering
from poly_csp.ordering.multi_opt import MultiOptSpec, run_multi_start_optimization

# Optional selector imports (keep pipeline runnable even before selector is implemented).
try:
    from poly_csp.chemistry.selectors import SelectorRegistry, SelectorTemplate
    from poly_csp.chemistry.functionalization import (
        apply_selector_pose_dihedrals,
        attach_selector,
    )
except (ImportError, ModuleNotFoundError):
    SelectorRegistry = None  # type: ignore
    SelectorTemplate = None  # type: ignore
    apply_selector_pose_dihedrals = None  # type: ignore
    attach_selector = None  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Unexpected error while importing selector modules.") from exc

# Optional MM imports (keep baseline path runnable if OpenMM is unavailable).
try:
    from poly_csp.mm.minimize import RelaxSpec, run_staged_relaxation
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
    multi_opt_enabled: bool = False
    multi_opt_rank: int = 0
    multi_opt_total_starts: int = 0
    multi_opt_seed_used: Optional[int] = None


def _cfg_to_helixspec(cfg: DictConfig) -> HelixSpec:
    return HelixSpec(
        name=str(cfg.helix.name),
        theta_rad=float(cfg.helix.theta_rad),
        rise_A=float(cfg.helix.rise_A),
        repeat_residues=int(cfg.helix.repeat_residues)
        if "repeat_residues" in cfg.helix
        else None,
        repeat_turns=int(cfg.helix.repeat_turns) if "repeat_turns" in cfg.helix else None,
        residues_per_turn=float(cfg.helix.residues_per_turn),
        pitch_A=float(cfg.helix.pitch_A),
        handedness=str(cfg.helix.handedness) if "handedness" in cfg.helix else "right",
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
    relax_cfg = cfg.relax if "relax" in cfg and cfg.relax is not None else {}
    anneal_cfg = (
        relax_cfg.anneal
        if hasattr(relax_cfg, "anneal") and relax_cfg.anneal is not None
        else {}
    )
    return RelaxSpec(
        enabled=bool(relax_cfg.enabled if "enabled" in relax_cfg else False),
        positional_k=float(
            relax_cfg.positional_k if "positional_k" in relax_cfg else 5000.0
        ),
        dihedral_k=float(relax_cfg.dihedral_k if "dihedral_k" in relax_cfg else 500.0),
        hbond_k=float(relax_cfg.hbond_k if "hbond_k" in relax_cfg else 50.0),
        mode=str(relax_cfg.mode if "mode" in relax_cfg else "geometry_pre_relax"),
        n_stages=int(relax_cfg.n_stages if "n_stages" in relax_cfg else 3),
        max_iterations=int(
            relax_cfg.max_iterations if "max_iterations" in relax_cfg else 200
        ),
        anneal_enabled=bool(anneal_cfg.enabled if "enabled" in anneal_cfg else False),
        t_start_K=float(anneal_cfg.t_start_K if "t_start_K" in anneal_cfg else 50.0),
        t_end_K=float(anneal_cfg.t_end_K if "t_end_K" in anneal_cfg else 350.0),
        anneal_steps=int(anneal_cfg.n_steps if "n_steps" in anneal_cfg else 2000),
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


def _selector_enabled(cfg: DictConfig) -> bool:
    return bool(
        "selector" in cfg
        and cfg.selector is not None
        and "enabled" in cfg.selector
        and cfg.selector.enabled
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

    polymer_kind: PolymerKind = str(cfg.polymer.kind)  # type: ignore[assignment]
    dp: int = int(cfg.polymer.dp)
    monomer_representation: MonomerRepresentation = str(
        cfg.polymer.monomer_representation
        if "monomer_representation" in cfg.polymer
        else "anhydro"
    )  # type: ignore[assignment]
    end_mode = str(cfg.polymer.end_mode if "end_mode" in cfg.polymer else "open")
    end_caps = (
        dict(OmegaConf.to_container(cfg.polymer.end_caps, resolve=True))
        if "end_caps" in cfg.polymer and cfg.polymer.end_caps is not None
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
    amber_parameter_backend = (
        str(cfg.amber.parameter_backend)
        if "amber" in cfg and cfg.amber is not None and "parameter_backend" in cfg.amber
        else "placeholder"
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

    relax_requested = bool(
        "relax" in cfg and cfg.relax is not None and "enabled" in cfg.relax and cfg.relax.enabled
    )
    relax_spec = _cfg_to_relax_spec(cfg)
    if relax_requested and (run_staged_relaxation is None or relax_spec is None):
        raise RuntimeError(
            "Relaxation requested but OpenMM modules are unavailable in this environment."
        )
    relax_mode = (
        str(relax_spec.mode)
        if (relax_spec is not None and bool(relax_spec.enabled))
        else "geometry_pre_relax"
    )

    if relax_mode == "ambertools_parameterized":
        if not amber_cfg_enabled:
            raise RuntimeError(
                "relax.mode=ambertools_parameterized requires amber.enabled=true."
            )
        if amber_parameter_backend.strip().lower() != "ambertools":
            raise RuntimeError(
                "relax.mode=ambertools_parameterized requires "
                "amber.parameter_backend=ambertools."
            )
        if "amber" not in output_export_formats:
            output_export_formats.append("amber")

    outdir = _ensure_outdir(
        cfg.output.dir if "output" in cfg and "dir" in cfg.output else "outputs"
    )

    # ---- Stage 1/2: backbone template + helical coords + polymerize.
    template = make_glucose_template(
        backbone.polymer,
        monomer_representation=backbone.monomer_representation,
    )
    coords = build_backbone_coords(template=template, helix=backbone.helix, dp=backbone.dp)

    mol_poly = polymerize(
        template=template,
        dp=backbone.dp,
        linkage="1-4",
        anomer="alpha" if backbone.polymer == "amylose" else "beta",
    )
    removed_old = (
        json.loads(mol_poly.GetProp("_poly_csp_removed_old_indices_json"))
        if mol_poly.HasProp("_poly_csp_removed_old_indices_json")
        else []
    )
    coords_for_mol = coords
    if removed_old:
        keep_mask = np.ones((coords.shape[0],), dtype=bool)
        keep_mask[np.asarray(removed_old, dtype=int)] = False
        coords_for_mol = coords[keep_mask]
    mol_poly = assign_conformer(mol_poly, coords_for_mol)
    mol_poly = apply_terminal_mode(
        mol=mol_poly,
        mode=backbone.end_mode,
        caps=backbone.end_caps,
        representation=backbone.monomer_representation,
    )

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

        selector_name = str(cfg.selector.name)
        selector_sites = [str(s) for s in cfg.selector.sites]  # type: ignore[assignment]
        selector = SelectorRegistry.get(selector_name)

        if "pose" in cfg.selector and cfg.selector.pose is not None:
            pose_payload = OmegaConf.to_container(cfg.selector.pose, resolve=True)
            if isinstance(pose_payload, dict):
                selector_pose = SelectorPoseSpec(**pose_payload)

        for res_i in range(backbone.dp):
            for site in selector_sites:
                mol_poly = attach_selector(
                    mol_polymer=mol_poly,
                    template=template,
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
    amber_export_done = False
    if relax_mode == "ambertools_parameterized":
        amber_summary = export_amber_artifacts(
            mol=mol_poly,
            outdir=outdir / amber_dir,
            model_name="model",
            charge_model=amber_charge_model,
            parameter_backend=amber_parameter_backend,
            net_charge=amber_net_charge,
        )
        amber_export_done = True

    # ---- Stage 5: optional restrained relaxation.
    relax_summary: dict[str, object] = {"enabled": False}
    relax_enabled = False
    if relax_spec is not None and bool(relax_spec.enabled):
        if run_staged_relaxation is None:
            raise RuntimeError(
                "Relaxation requested but run_staged_relaxation is unavailable."
            )
        mol_poly, relax_summary = run_staged_relaxation(
            mol=mol_poly,
            spec=relax_spec,
            selector=selector,
            amber_artifacts=amber_summary if relax_mode == "ambertools_parameterized" else None,
        )
        relax_enabled = True

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

    # ---- Stage 8: optional AMBER export (or already exported for parameterized relaxation).
    amber_enabled = "amber" in output_export_formats or amber_export_done
    if amber_enabled and not amber_export_done:
        amber_summary = export_amber_artifacts(
            mol=mol_poly,
            outdir=outdir / amber_dir,
            model_name="model",
            charge_model=amber_charge_model,
            parameter_backend=amber_parameter_backend,
            net_charge=amber_net_charge,
        )
        amber_export_done = True

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
        relax_enabled=bool(relax_enabled),
        relax_mode=relax_mode,
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
        write_pdb_from_rdkit(mol_poly, pdb_path)

    sdf_path = outdir / "model.sdf"
    if "sdf" in output_export_formats:
        write_sdf(mol_poly, sdf_path)

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
            if "pdb" in output_export_formats:
                write_pdb_from_rdkit(result.mol, rank_dir / "model.pdb")
            if "sdf" in output_export_formats:
                write_sdf(result.mol, rank_dir / "model.sdf")
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
