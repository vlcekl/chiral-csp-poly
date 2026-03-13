from __future__ import annotations

from poly_csp.config.schema import RuntimeForcefieldOptions
from poly_csp.forcefield.relaxation import RelaxSpec
from poly_csp.ordering.optimize import OrderingSpec
from poly_csp.pipelines.build_csp import _cfg_to_relax_spec, _relax_spec_with_enabled


def test_runtime_forcefield_options_expose_seed_bias_fields() -> None:
    options = RuntimeForcefieldOptions(
        anti_stacking_sigma_scale=1.35,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.19,
        hbond_neighbor_window=2,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="donor_heavy",
        skip_full_stage=True,
    )

    assert options.anti_stacking_sigma_scale == 1.35
    assert options.soft_exclude_14 is True
    assert options.ideal_hbond_target_nm == 0.19
    assert options.hbond_neighbor_window == 2
    assert options.hbond_pairing_mode == "nearest_unique"
    assert options.hbond_restraint_atom_mode == "donor_heavy"
    assert options.skip_full_stage is True


def test_ordering_spec_exposes_seed_bias_fields() -> None:
    spec = OrderingSpec(
        enabled=True,
        hbond_k=25.0,
        anti_stacking_sigma_scale=1.2,
        soft_exclude_14=True,
        ideal_hbond_target_nm=0.18,
        hbond_pairing_mode="nearest_unique",
        hbond_restraint_atom_mode="hydrogen_if_present",
        skip_full_stage=True,
    )

    assert spec.enabled is True
    assert spec.hbond_k == 25.0
    assert spec.anti_stacking_sigma_scale == 1.2
    assert spec.soft_exclude_14 is True
    assert spec.ideal_hbond_target_nm == 0.18
    assert spec.hbond_pairing_mode == "nearest_unique"
    assert spec.hbond_restraint_atom_mode == "hydrogen_if_present"
    assert spec.skip_full_stage is True


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
