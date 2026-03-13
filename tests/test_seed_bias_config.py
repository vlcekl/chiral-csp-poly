from __future__ import annotations

from omegaconf import OmegaConf

from poly_csp.config.schema import RuntimeForcefieldOptions
from poly_csp.config.schema import SeedBiasOptions
from poly_csp.config.schema import SoftSelectorHbondBiasOptions
from poly_csp.forcefield.relaxation import RelaxSpec
from poly_csp.ordering.optimize import OrderingSpec
from poly_csp.pipelines.build_csp import (
    _cfg_to_forcefield_options,
    _cfg_to_ordering_spec,
    _cfg_to_relax_spec,
    _relax_spec_with_enabled,
)


def test_runtime_forcefield_options_expose_seed_bias_fields() -> None:
    options = RuntimeForcefieldOptions(
        soft_repulsion_k_kj_per_mol_nm2=900.0,
        soft_repulsion_cutoff_nm=0.65,
        anti_stacking_sigma_scale=1.35,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.19,
        hbond_neighbor_window=2,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="donor_heavy",
        skip_full_stage=True,
        soft_selector_hbond_bias=SoftSelectorHbondBiasOptions(
            enabled=True,
            epsilon_kj_per_mol=3.2,
            r0_nm=0.205,
            half_width_nm=0.045,
            hbond_neighbor_window=2,
        ),
    )

    assert options.soft_repulsion_k_kj_per_mol_nm2 == 900.0
    assert options.soft_repulsion_cutoff_nm == 0.65
    assert options.anti_stacking_sigma_scale == 1.35
    assert options.soft_exclude_14 is True
    assert options.ideal_hbond_target_nm == 0.19
    assert options.hbond_neighbor_window == 2
    assert options.hbond_pairing_mode == "nearest_unique"
    assert options.hbond_restraint_atom_mode == "donor_heavy"
    assert options.skip_full_stage is True
    assert options.soft_selector_hbond_bias.enabled is True
    assert options.soft_selector_hbond_bias.epsilon_kj_per_mol == 3.2


def test_ordering_spec_exposes_seed_bias_fields() -> None:
    spec = OrderingSpec(
        enabled=True,
        soft_repulsion_k_kj_per_mol_nm2=925.0,
        soft_repulsion_cutoff_nm=0.66,
        hbond_k=25.0,
        anti_stacking_sigma_scale=1.2,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.18,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="hydrogen_if_present",
        skip_full_stage=True,
        soft_selector_hbond_bias=SoftSelectorHbondBiasOptions(
            enabled=True,
            epsilon_kj_per_mol=3.4,
            r0_nm=0.20,
            half_width_nm=0.05,
            hbond_neighbor_window=1,
        ),
    )

    assert spec.enabled is True
    assert spec.soft_repulsion_k_kj_per_mol_nm2 == 925.0
    assert spec.soft_repulsion_cutoff_nm == 0.66
    assert spec.hbond_k == 25.0
    assert spec.anti_stacking_sigma_scale == 1.2
    assert spec.soft_exclude_14 is True
    assert spec.ideal_hbond_target_nm == 0.18
    assert spec.hbond_pairing_mode == "nearest_unique"
    assert spec.hbond_restraint_atom_mode == "hydrogen_if_present"
    assert spec.skip_full_stage is True
    assert spec.soft_selector_hbond_bias.enabled is True
    assert spec.soft_selector_hbond_bias.epsilon_kj_per_mol == 3.4


def test_cfg_to_relax_spec_maps_seed_bias_fields() -> None:
    options = RuntimeForcefieldOptions(
        enabled=True,
        relax_enabled=True,
        anti_stacking_sigma_scale=1.25,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.20,
        hbond_neighbor_window=3,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="donor_heavy",
        skip_full_stage=True,
        soft_selector_hbond_bias=SoftSelectorHbondBiasOptions(
            enabled=True,
            epsilon_kj_per_mol=2.8,
            r0_nm=0.20,
            half_width_nm=0.05,
            hbond_neighbor_window=2,
        ),
    )

    relax = _cfg_to_relax_spec(options)

    assert isinstance(relax, RelaxSpec)
    assert relax.enabled is True
    assert relax.anti_stacking_sigma_scale == 1.25
    assert relax.soft_exclude_14 is True
    assert relax.ideal_hbond_target_nm == 0.20
    assert relax.hbond_neighbor_window == 3
    assert relax.hbond_pairing_mode == "nearest_unique"
    assert relax.hbond_restraint_atom_mode == "donor_heavy"
    assert relax.skip_full_stage is True
    assert relax.soft_selector_hbond_bias.enabled is True
    assert relax.soft_selector_hbond_bias.epsilon_kj_per_mol == 2.8


def test_seed_bias_options_expose_shared_fields() -> None:
    seed_bias = SeedBiasOptions(
        soft_repulsion_k_kj_per_mol_nm2=900.0,
        soft_repulsion_cutoff_nm=0.65,
        anti_stacking_sigma_scale=1.4,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.19,
        hbond_neighbor_window=2,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="hydrogen_if_present",
        skip_full_stage=True,
        soft_selector_hbond_bias=SoftSelectorHbondBiasOptions(
            enabled=True,
            epsilon_kj_per_mol=3.0,
            r0_nm=0.20,
            half_width_nm=0.05,
            hbond_neighbor_window=1,
        ),
    )

    assert seed_bias.soft_repulsion_k_kj_per_mol_nm2 == 900.0
    assert seed_bias.soft_repulsion_cutoff_nm == 0.65
    assert seed_bias.anti_stacking_sigma_scale == 1.4
    assert seed_bias.soft_exclude_14 is True
    assert seed_bias.ideal_hbond_target_nm == 0.19
    assert seed_bias.hbond_neighbor_window == 2
    assert seed_bias.hbond_pairing_mode == "nearest_unique"
    assert seed_bias.hbond_restraint_atom_mode == "hydrogen_if_present"
    assert seed_bias.skip_full_stage is True
    assert seed_bias.soft_selector_hbond_bias.enabled is True


def test_cfg_to_ordering_spec_uses_shared_seed_bias_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "seed_bias": {
                "soft_repulsion_k_kj_per_mol_nm2": 910.0,
                "soft_repulsion_cutoff_nm": 0.64,
                "anti_stacking_sigma_scale": 1.38,
                "soft_exclude_14": True,
                "ideal_hbond_target_nm": 0.19,
                "hbond_neighbor_window": 2,
                "hbond_pairing_mode": "nearest_unique",
                "hbond_restraint_atom_mode": "hydrogen_if_present",
                "skip_full_stage": True,
                "soft_selector_hbond_bias": {
                    "enabled": True,
                    "epsilon_kj_per_mol": 3.3,
                    "r0_nm": 0.205,
                    "half_width_nm": 0.045,
                    "hbond_neighbor_window": 2,
                },
            },
            "ordering": {
                "enabled": True,
                "hbond_k": 0.0,
            },
        }
    )

    spec = _cfg_to_ordering_spec(cfg)

    assert spec.enabled is True
    assert spec.hbond_k == 0.0
    assert spec.soft_repulsion_k_kj_per_mol_nm2 == 910.0
    assert spec.soft_repulsion_cutoff_nm == 0.64
    assert spec.anti_stacking_sigma_scale == 1.38
    assert spec.soft_exclude_14 is True
    assert spec.ideal_hbond_target_nm == 0.19
    assert spec.hbond_neighbor_window == 2
    assert spec.hbond_pairing_mode == "nearest_unique"
    assert spec.hbond_restraint_atom_mode == "hydrogen_if_present"
    assert spec.skip_full_stage is True
    assert spec.soft_selector_hbond_bias.enabled is True
    assert spec.soft_selector_hbond_bias.hbond_neighbor_window == 2


def test_cfg_to_forcefield_options_uses_shared_seed_bias_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "seed_bias": {
                "soft_repulsion_k_kj_per_mol_nm2": 920.0,
                "soft_repulsion_cutoff_nm": 0.67,
                "anti_stacking_sigma_scale": 1.4,
                "soft_exclude_14": True,
                "ideal_hbond_target_nm": 0.19,
                "hbond_neighbor_window": 2,
                "hbond_pairing_mode": "nearest_unique",
                "hbond_restraint_atom_mode": "hydrogen_if_present",
                "skip_full_stage": True,
                "soft_selector_hbond_bias": {
                    "enabled": True,
                    "epsilon_kj_per_mol": 2.9,
                    "r0_nm": 0.20,
                    "half_width_nm": 0.05,
                    "hbond_neighbor_window": 1,
                },
            },
            "forcefield": {
                "options": {
                    "enabled": True,
                    "relax_enabled": False,
                }
            },
        }
    )

    options = _cfg_to_forcefield_options(cfg)

    assert options.enabled is True
    assert options.relax_enabled is False
    assert options.soft_repulsion_k_kj_per_mol_nm2 == 920.0
    assert options.soft_repulsion_cutoff_nm == 0.67
    assert options.anti_stacking_sigma_scale == 1.4
    assert options.soft_exclude_14 is True
    assert options.ideal_hbond_target_nm == 0.19
    assert options.hbond_neighbor_window == 2
    assert options.hbond_pairing_mode == "nearest_unique"
    assert options.hbond_restraint_atom_mode == "hydrogen_if_present"
    assert options.skip_full_stage is True
    assert options.soft_selector_hbond_bias.enabled is True
    assert options.soft_selector_hbond_bias.epsilon_kj_per_mol == 2.9


def test_relax_spec_with_enabled_preserves_seed_bias_fields() -> None:
    options = RuntimeForcefieldOptions(
        enabled=True,
        relax_enabled=False,
        anti_stacking_sigma_scale=1.3,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.19,
        hbond_neighbor_window=2,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="donor_heavy",
        skip_full_stage=True,
        soft_selector_hbond_bias=SoftSelectorHbondBiasOptions(
            enabled=True,
            epsilon_kj_per_mol=3.1,
            r0_nm=0.20,
            half_width_nm=0.05,
            hbond_neighbor_window=1,
        ),
    )

    relax = _relax_spec_with_enabled(options, enabled=True)

    assert isinstance(relax, RelaxSpec)
    assert relax.enabled is True
    assert relax.anti_stacking_sigma_scale == 1.3
    assert relax.soft_exclude_14 is True
    assert relax.ideal_hbond_target_nm == 0.19
    assert relax.hbond_neighbor_window == 2
    assert relax.hbond_pairing_mode == "nearest_unique"
    assert relax.hbond_restraint_atom_mode == "donor_heavy"
    assert relax.skip_full_stage is True
    assert relax.soft_selector_hbond_bias.enabled is True
    assert relax.soft_selector_hbond_bias.epsilon_kj_per_mol == 3.1
