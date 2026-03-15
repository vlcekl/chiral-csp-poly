# Helix Literature Parameter Integration Plan

## Findings

### 1. `axial_repeat_A` is already wired

The updated derivatized helix presets will already affect the CSP build without code changes.

Reason:

- [HelixSpec](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/config/schema.py) derives `rise_A` and `pitch_A` from `repeat_residues`, `repeat_turns`, and `axial_repeat_A`
- [build_csp.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/pipelines/build_csp.py) loads the helix preset directly into `HelixSpec`
- [backbone_builder.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/structure/backbone_builder.py) then uses the resulting `theta_rad` / `rise_A` and stores the updated helix metadata on the built molecule

So the new values:

- amylose: `axial_repeat_A = 15.614`
- cellulose: `axial_repeat_A = 15.3`

should already change:

- `rise_A`
- `pitch_A`
- periodic box length / axial repeat in periodic builds
- the stored helix metadata used by symmetry ordering

### 2. The literature `phi/psi` values are not wired yet

The current helix presets use:

- `bb_phi`
- `bb_psi`

but the schema currently defines:

- `glycosidic_phi_deg`
- `glycosidic_psi_deg`
- `glycosidic_omega_deg`

and there are no code paths that currently consume any of those torsion fields. So the new literature torsion targets are currently reference-only and have no runtime effect.

### 3. The literature `phi/psi` values do match the current backbone DOF definitions

Current code definitions in [symmetry_opt.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/ordering/symmetry_opt.py):

- `bb_phi = O5(i)-C1(i)-O4(i+1)-C4(i+1)`
- `bb_psi = C3(i+1)-C4(i+1)-O4(i+1)-C1(i)`

The usual glycosidic convention is:

- `phi = O5(i)-C1(i)-O4(i+1)-C4(i+1)`
- `psi = C1(i)-O4(i+1)-C4(i+1)-C3(i+1)`

`bb_phi` is identical to the standard convention.

`bb_psi` is the fully reversed atom order of the standard convention, which gives the same dihedral angle. So the literature values should map directly onto the current `bb_phi` / `bb_psi` DOFs.

## What needs to be adjusted

### No code change required

- the new `axial_repeat_A` values themselves

### Code or config adjustment required

For the literature torsion targets, one of these must happen:

1. Rename the helix-config keys from `bb_phi` / `bb_psi` to `glycosidic_phi_deg` / `glycosidic_psi_deg`
2. Or add schema aliases so `bb_phi` / `bb_psi` are accepted as alternate input names

That only makes the values loadable. A second step is still required to make them affect ordering.

## Recommended implementation

### Phase 1: load and persist the literature torsion targets

Adjust [HelixSpec](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/config/schema.py) to accept the literature torsion fields cleanly.

Recommended approach:

- keep the canonical schema names as `glycosidic_phi_deg` / `glycosidic_psi_deg`
- accept `bb_phi` / `bb_psi` as input aliases for backwards-compatible config loading

Then persist those values on the built molecule in [backbone_builder.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/structure/backbone_builder.py), alongside the existing stored helix metadata, for example:

- `_poly_csp_helix_glycosidic_phi_deg`
- `_poly_csp_helix_glycosidic_psi_deg`

### Phase 2: expose the targets in reports and ordering summaries

Add them to:

- `build_report.json`
- `ordering_summary`

so a run clearly shows:

- literature target `phi/psi`
- initial measured `bb_phi/bb_psi`
- final measured `bb_phi/bb_psi`
- deviation from literature targets

This is useful even before they actively influence the optimization.

### Phase 3: make them available to ordering as optional targets

Do not hard-wire them as mandatory restraints immediately.

Recommended first use:

- initialize periodic/open-chain backbone refinement around the literature values
- optionally add a weak quadratic target term for `bb_phi` / `bb_psi` in the backbone-refinement objective
- keep it configurable and off by default until benchmarked

This is lower-risk than forcing the literature values directly from the start.

## Test updates needed

The new axial repeats mean several tests with hard-coded derivatized helix expectations must be updated, including:

- [tests/test_backbone_helical_coords.py](/home/lukas/work/projects/chiral_csp_poly/tests/test_backbone_helical_coords.py)
- [tests/test_pipeline_config_groups.py](/home/lukas/work/projects/chiral_csp_poly/tests/test_pipeline_config_groups.py)
- [tests/test_pipeline_glycam_runtime.py](/home/lukas/work/projects/chiral_csp_poly/tests/test_pipeline_glycam_runtime.py)

Any test expecting derivatized amylose `14.6 / 3.65` or derivatized cellulose `16.2 / 5.4` will need to be updated to the new literature-aligned values.
