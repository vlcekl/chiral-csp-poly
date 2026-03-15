"""Reference helix presets.

This module is reference-only and is not the runtime source of truth for
pipeline defaults. Active defaults are managed by Hydra YAML files in
`conf/structure/helix/`.
"""

from __future__ import annotations

from poly_csp.config.schema import HelixSpec


def _make_from_repeat(
    name: str,
    repeat_residues: int,
    repeat_turns: int,
    axial_repeat_A: float,
    handedness: str = "right",
    glycosidic_phi_deg: float | None = None,
    glycosidic_psi_deg: float | None = None,
) -> HelixSpec:
    return HelixSpec(
        name=name,
        repeat_residues=repeat_residues,
        repeat_turns=repeat_turns,
        axial_repeat_A=axial_repeat_A,
        handedness=handedness,
        glycosidic_phi_deg=glycosidic_phi_deg,
        glycosidic_psi_deg=glycosidic_psi_deg,
    )


# --- Cellulose I reference: 2_1 screw (180° per residue), ~10.3 Å repeat per 2 residues.
cellulose_natural_i_2_1 = _make_from_repeat(
    name="cellulose_I_2_1_natural",
    repeat_residues=2,
    repeat_turns=1,
    axial_repeat_A=10.3,
    handedness="right",  # 2_1 is effectively non-chiral; choose a sign convention and stick to it.
)

# --- Derivatized amylose CSP: left-handed 4/3 helix, 15.614 A per repeat.
amylose_csp_4_3_derivatized = _make_from_repeat(
    name="amylose_CSP_4_3_derivatized",
    repeat_residues=4,
    repeat_turns=3,
    axial_repeat_A=15.614,
    handedness="left",
    glycosidic_phi_deg=-68.5,
    glycosidic_psi_deg=-42.0,
)

# --- Derivatized cellulose CSP: left-handed 3/2 helix, 15.3 A per repeat.
cellulose_csp_3_2_derivatized = _make_from_repeat(
    name="cellulose_CSP_3_2_derivatized",
    repeat_residues=3,
    repeat_turns=2,
    axial_repeat_A=15.3,
    handedness="left",
    glycosidic_phi_deg=60.0,
    glycosidic_psi_deg=0.0,
)

# --- Amylose V-helix: 6 residues per turn, pitch ~7.8–7.9 Å.
amylose_v6_1 = _make_from_repeat(
    name="amylose_V_6_1",
    repeat_residues=6,
    repeat_turns=1,
    axial_repeat_A=7.8,
    handedness="left",
)
