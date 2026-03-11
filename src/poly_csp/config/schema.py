# poly_csp/config/schema.py
from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple
from pydantic import BaseModel, Field, PositiveInt, confloat, model_validator


PolymerKind = Literal["amylose", "cellulose"]
Site = Literal["C2", "C3", "C6"]
Handedness = Literal["left", "right"]
MonomerRepresentation = Literal["anhydro", "natural_oh"]
EndMode = Literal["open", "capped", "periodic"]


class HelixSpec(BaseModel):
    """
    Defines a helix by a screw operation applied once per residue:
      - rotate about +z by theta_rad
      - translate along +z by rise_A
    """
    name: str

    # Screw parameters (per residue)
    theta_rad: Optional[float] = Field(
        None,
        description="Rotation per residue about +z (radians).",
    )
    rise_A: Optional[float] = Field(
        None,
        description="Translation per residue along +z (angstrom).",
    )

    # For tight helices like 4/3, store the rational form explicitly.
    repeat_residues: Optional[PositiveInt] = Field(
        None,
        description="Residues in helical repeat (e.g., 4).",
    )
    repeat_turns: Optional[PositiveInt] = Field(
        None,
        description="Turns in helical repeat (e.g., 3).",
    )

    # Informational/derived convenience fields
    residues_per_turn: Optional[confloat(gt=0)] = Field(
        None,
        description="n = residues per 360° turn.",
    )
    pitch_A: Optional[float] = Field(
        None,
        description="Pitch (angstrom) per 360° turn.",
    )
    axial_repeat_A: Optional[float] = Field(
        None,
        description=(
            "Axial translation (angstrom) across the full rational helical repeat, "
            "e.g. 14.6 A for a 4-residue/3-turn amylose CSP repeat."
        ),
    )

    handedness: Handedness = "right"
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    inter_rod_distance_A: Optional[float] = None
    core_diameter_A: Optional[float] = None
    reference_label: Optional[str] = None
    reference_columns: Tuple[str, ...] = Field(default_factory=tuple)

    # Optional: torsion targets to keep polymerizable geometry
    # Use degrees here for human readability; convert internally.
    # (You’ll refine these once you lock your monomer atom labels.)
    glycosidic_phi_deg: Optional[float] = None
    glycosidic_psi_deg: Optional[float] = None
    glycosidic_omega_deg: Optional[float] = None

    @model_validator(mode="after")
    def _normalize(self) -> "HelixSpec":
        tol = 1e-8
        fields_set = set(self.model_fields_set)

        def _coerce(name: str) -> float | None:
            value = getattr(self, name)
            return None if value is None else float(value)

        def _set_or_check(name: str, derived: float) -> float:
            current = _coerce(name)
            if current is None:
                setattr(self, name, float(derived))
                return float(derived)
            if not math.isclose(current, float(derived), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(
                    f"HelixSpec field {name!r}={current} is inconsistent with derived value "
                    f"{float(derived)} for helix {self.name!r}."
                )
            return current

        handedness = self.handedness
        theta = _coerce("theta_rad")
        if theta is not None and abs(theta) > tol:
            theta_handedness: Handedness = "left" if theta < 0.0 else "right"
            if "handedness" in fields_set and handedness != theta_handedness:
                raise ValueError(
                    f"HelixSpec handedness {handedness!r} is inconsistent with "
                    f"theta_rad={theta} for helix {self.name!r}."
                )
            handedness = theta_handedness
            self.handedness = handedness

        repeat_residues = (
            int(self.repeat_residues) if self.repeat_residues is not None else None
        )
        repeat_turns = int(self.repeat_turns) if self.repeat_turns is not None else None
        if (repeat_residues is None) ^ (repeat_turns is None):
            raise ValueError(
                "repeat_residues and repeat_turns must be provided together when using "
                "repeat-based helix metadata."
            )

        if repeat_residues is not None and repeat_turns is not None:
            residues_per_turn = _set_or_check(
                "residues_per_turn",
                float(repeat_residues) / float(repeat_turns),
            )
            _set_or_check(
                "theta_rad",
                (-1.0 if handedness == "left" else 1.0)
                * (2.0 * math.pi * float(repeat_turns) / float(repeat_residues)),
            )

            axial_repeat = _coerce("axial_repeat_A")
            rise_A = _coerce("rise_A")
            pitch_A = _coerce("pitch_A")

            if axial_repeat is None and rise_A is not None:
                axial_repeat = float(rise_A) * float(repeat_residues)
            if axial_repeat is None and pitch_A is not None:
                axial_repeat = float(pitch_A) * float(repeat_turns)

            if axial_repeat is not None:
                axial_repeat = _set_or_check("axial_repeat_A", axial_repeat)
                rise_A = _set_or_check(
                    "rise_A",
                    float(axial_repeat) / float(repeat_residues),
                )
                pitch_A = _set_or_check(
                    "pitch_A",
                    float(axial_repeat) / float(repeat_turns),
                )
            else:
                if rise_A is None and pitch_A is None:
                    raise ValueError(
                        f"HelixSpec {self.name!r} needs either axial_repeat_A, rise_A, "
                        "or pitch_A when repeat_residues/repeat_turns are provided."
                    )
                if rise_A is not None:
                    rise_A = _set_or_check("rise_A", rise_A)
                    pitch_A = _set_or_check(
                        "pitch_A",
                        float(rise_A) * float(residues_per_turn),
                    )
                    _set_or_check(
                        "axial_repeat_A",
                        float(rise_A) * float(repeat_residues),
                    )
                else:
                    pitch_A = _set_or_check("pitch_A", float(pitch_A))
                    rise_A = _set_or_check(
                        "rise_A",
                        float(pitch_A) / float(residues_per_turn),
                    )
                    _set_or_check(
                        "axial_repeat_A",
                        float(pitch_A) * float(repeat_turns),
                    )
        else:
            residues_per_turn = _coerce("residues_per_turn")
            rise_A = _coerce("rise_A")
            pitch_A = _coerce("pitch_A")

            if residues_per_turn is None and theta is not None:
                if abs(theta) <= tol:
                    raise ValueError(
                        f"HelixSpec {self.name!r} cannot derive residues_per_turn from "
                        "theta_rad=0. Provide residues_per_turn explicitly."
                    )
                residues_per_turn = _set_or_check(
                    "residues_per_turn",
                    (2.0 * math.pi) / abs(float(theta)),
                )
            if residues_per_turn is None:
                raise ValueError(
                    f"HelixSpec {self.name!r} is missing residues_per_turn."
                )
            if theta is None:
                theta = _set_or_check(
                    "theta_rad",
                    (-1.0 if handedness == "left" else 1.0)
                    * ((2.0 * math.pi) / float(residues_per_turn)),
                )
            if rise_A is None and pitch_A is None:
                raise ValueError(
                    f"HelixSpec {self.name!r} needs at least one of rise_A or pitch_A."
                )
            if rise_A is not None:
                rise_A = _set_or_check("rise_A", rise_A)
                _set_or_check(
                    "pitch_A",
                    float(rise_A) * float(residues_per_turn),
                )
            else:
                pitch_A = _set_or_check("pitch_A", float(pitch_A))
                _set_or_check(
                    "rise_A",
                    float(pitch_A) / float(residues_per_turn),
                )

        if self.theta_rad is None or self.rise_A is None:
            raise ValueError(f"HelixSpec {self.name!r} is missing screw parameters.")
        if self.residues_per_turn is None or self.pitch_A is None:
            raise ValueError(
                f"HelixSpec {self.name!r} is missing derived pitch metadata."
            )

        return self


class BackboneSpec(BaseModel):
    polymer: PolymerKind
    dp: PositiveInt
    monomer_representation: MonomerRepresentation = "anhydro"
    end_mode: EndMode = "open"
    end_caps: Dict[str, str] = Field(default_factory=dict)
    helix: HelixSpec
    # ring pucker defaults to 4C1; use later if you add internal coordinate enforcement
    ring_pucker: Literal["4C1"] = "4C1"


class SelectorPoseSpec(BaseModel):
    """
    Deterministic initial pose rules for a selector in the residue-local frame.
    Keep minimal here; expand later.
    """
    # Example: initial dihedrals to apply after bonding (degrees)
    dihedral_targets_deg: Dict[str, float] = Field(default_factory=dict)

    # Optional directional preferences in residue-local frame
    # (unit vectors in local frame; use later if needed)
    carbonyl_dir_local: Optional[Tuple[float, float, float]] = None
    aromatic_normal_local: Optional[Tuple[float, float, float]] = None


class SelectorRuntimeSpec(BaseModel):
    enabled: bool = False
    name: Optional[str] = None
    sites: Tuple[Site, ...] = ("C2", "C3", "C6")
    pose: SelectorPoseSpec = Field(default_factory=SelectorPoseSpec)

    @model_validator(mode="after")
    def _validate_enabled_selector_has_name(self) -> "SelectorRuntimeSpec":
        if self.enabled and (self.name is None or not str(self.name).strip()):
            raise ValueError("Enabled selector runtime config requires a selector name.")
        return self


class PhasePresetSpec(BaseModel):
    column_id: str
    phase_name: str
    manufacturer: Optional[str] = None
    chemical_name: Optional[str] = None
    attachment_mode: Literal["coated", "immobilized"]
    attachment_description: Optional[str] = None
    silica_tether_description: Optional[str] = None


class ScalingPair(BaseModel):
    scee: confloat(gt=0) = 1.0
    scnb: confloat(gt=0) = 1.0


class MixingRules(BaseModel):
    backbone_backbone: ScalingPair = Field(
        default_factory=lambda: ScalingPair(scee=1.0, scnb=1.0)
    )
    selector_selector: ScalingPair = Field(
        default_factory=lambda: ScalingPair(scee=1.2, scnb=2.0)
    )
    cross_boundary: ScalingPair = Field(
        default_factory=lambda: ScalingPair(scee=1.0, scnb=1.0)
    )


class AnnealOptions(BaseModel):
    enabled: bool = False
    t_start_K: float = 50.0
    t_end_K: float = 350.0
    n_steps: PositiveInt = 2000
    cool_down: bool = True


class RuntimeForcefieldOptions(BaseModel):
    enabled: bool = False
    relax_enabled: bool = False
    cache_enabled: bool = True
    cache_dir: Optional[str] = None
    positional_k: float = 5000.0
    dihedral_k: float = 500.0
    hbond_k: float = 50.0
    freeze_backbone: bool = True
    soft_n_stages: PositiveInt = 3
    soft_max_iterations: PositiveInt = 200
    full_max_iterations: PositiveInt = 200
    final_restraint_factor: float = 0.15
    soft_repulsion_k_kj_per_mol_nm2: float = 800.0
    soft_repulsion_cutoff_nm: confloat(gt=0) = 0.6
    anneal: AnnealOptions = Field(default_factory=AnnealOptions)


class ForceFieldConfig(BaseModel):
    options: RuntimeForcefieldOptions = Field(default_factory=RuntimeForcefieldOptions)
    mixing_rules: MixingRules = Field(default_factory=MixingRules)
