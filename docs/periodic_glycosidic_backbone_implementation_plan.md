# Periodic Glycosidic Backbone Implementation Plan

## Problem

The current `symmetry_network` backbone refinement only exposes glycosidic `bb_phi` and `bb_psi` on open-chain ordering models. In the production CSP pipeline, ordering is usually run on a periodic model because `topology.backbone.end_mode=periodic`. `periodic_handoff.enabled` only affects the later export/handoff path; it does not change the ordering model.

That means the current periodic `symmetry_network` search has:

- selector DOFs: `tau_link`, `tau_ar`, `tau_attach` for each active site
- backbone DOFs: only `bb_c6_omega`

The latest `build_report.json` confirms this:

- `end_mode = "periodic"`
- `active_backbone_dihedral_names = ["bb_c6_omega"]`
- `backbone_refinement_applied = true`
- `backbone_refinement_selected = false`

This is consistent with the persistent `1/0/0` H-bond-family result. The optimizer can realize the `C2 -> C3` zipper with selector DOFs alone, but it never gets the periodic glycosidic freedom needed to search for `C3 -> C2` and the `C6` pitch bridge on the actual periodic ordering model.

## Goal

Add periodic-safe glycosidic backbone DOFs to the `symmetry_network` backbone refinement stage so that periodic CSP ordering can optimize:

- `bb_phi_periodic`
- `bb_psi_periodic`
- `bb_c6_omega`

while preserving:

- exact screw symmetry
- a valid periodic closure bond
- the existing selector-side `symmetry_network` search and scoring path

## Design Choice

### Rejected: direct wrapped-cell bond rotations

The current open-chain implementation rotates either the downstream suffix or the upstream prefix around a glycosidic bond. That works because the chain has free ends.

For a periodic chain, the closure linkage connects residue `dp - 1` to residue `0` across the cell boundary. A direct wrapped-cell rotation is not well defined because:

- there is no true free suffix or prefix
- the closure linkage uses an image of residue `0`, not the in-cell coordinates
- naively rotating atoms in the base cell can break closure or destroy periodic equivalence

### Recommended: lifted periodic backbone representation

Implement periodic glycosidic refinement on a temporary lifted chain representation:

1. Build an unwrapped ordered residue series for one full periodic cell plus one image residue.
2. Treat the closure linkage as the normal linkage between residue `dp - 1` and image residue `dp`.
3. Apply global periodic `phi` and `psi` to that lifted chain with deterministic masks.
4. Fold residues `0..dp-1` back into the base cell and overwrite the periodic ordering coordinates.

This keeps the implementation local to the backbone refinement stage and avoids rewriting the main selector-side symmetry engine.

## Backbone DOF Definition

Use one global periodic angle per torsion family, not one per linkage:

- `bb_phi_periodic`: `O5(i)-C1(i)-O4(i+1)-C4(i+1)` for all `i = 0 .. dp-1`
- `bb_psi_periodic`: `C3(i+1)-C4(i+1)-O4(i+1)-C1(i)` for all `i = 0 .. dp-1`
- `bb_c6_omega`: unchanged

`i + 1` is interpreted on the lifted chain, so the closure linkage is represented by residue `dp - 1` to image residue `dp`.

These are still symmetry-coupled DOFs:

- one scalar `phi`
- one scalar `psi`
- one scalar `omega`

not per-residue independent torsions.

## Recommended Implementation

### 1. Add a periodic backbone refinement engine

In [src/poly_csp/ordering/symmetry_opt.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/ordering/symmetry_opt.py), add a periodic-specific backbone refinement path rather than trying to force the current open-chain mask logic to handle closure.

Add helpers for:

- collecting residue-core atom index maps for all residues
- building lifted coordinates for residues `0..dp` using the stored screw transform
- mapping lifted residue `dp` back to the periodic image of residue `0`
- folding the refined lifted coordinates back into residues `0..dp-1`

### 2. Implement periodic `phi` / `psi` application on lifted coordinates

Add periodic counterparts to the current open-chain helpers:

- `_periodic_phi_coupled_dihedral(...)`
- `_periodic_psi_coupled_dihedral(...)`

But instead of storing only atom indices and masks into the base-cell coordinate array, store lifted-chain instances that are applied inside a periodic backbone builder.

The minimal-disruption approach is:

- keep the current `_SymmetryBackboneRefinementEngine` for open chains
- add a small periodic variant that overrides `build_coords()`
- reuse the current DE objective, scoring, and summary/report machinery

### 3. Update engine selection

In `_build_backbone_refinement_engine(...)`:

- if the ordering model is open-chain, keep the current behavior
- if the ordering model is periodic and `C2` or `C3` are active, build the periodic refinement engine instead of returning `periodic_glycosidic_backbone_unsupported`

No new user-facing config is needed for the first iteration. The existing flags should keep the same meaning:

- `symmetry_backbone_refine_enabled`
- `symmetry_backbone_include_phi`
- `symmetry_backbone_include_psi`
- `symmetry_backbone_include_c6_omega`

## Algorithm

For each periodic refinement candidate:

1. Start from the current best selector-stage periodic coordinates.
2. Lift the backbone residues into an unwrapped chain with one extra image residue.
3. Apply the candidate periodic `phi` / `psi` values across all glycosidic linkages on the lifted chain.
4. Apply `bb_c6_omega` on the base-cell residues as today.
5. Re-embed the refined backbone coordinates into the base cell.
6. Reapply exact screw projection for selector blocks if selector re-optimization is enabled.
7. Score the full periodic polymer with the existing `symmetry_network` cleanup-stage objective.

## Validation

Add tests for:

- periodic `C2`-only refinement activates `bb_phi_periodic` and `bb_psi_periodic`
- periodic `C2/C3/C6` refinement reports all expected backbone DOFs
- periodic closure bond length remains within tolerance after backbone refinement
- selector screw symmetry RMSD remains near zero after periodic backbone refinement
- a periodic amylose regression fixture no longer reports `active_backbone_dihedral_names = ["bb_c6_omega"]` when `C2/C3` are active

Benchmark on both:

- amylose CSP periodic unit cell
- cellulose CSP periodic unit cell

Cellulose should be used as a comparison case, not as a substitute for the implementation. It may show a different basin, but it does not remove the missing periodic glycosidic DOF problem.

## Rollout

### Phase 1

Implement periodic `phi` / `psi` support in the backbone refinement stage only. Do not change the primary selector-side `network_capture` / `network_cleanup` stages.

### Phase 2

Benchmark whether periodic glycosidic refinement moves the family metrics beyond `1/0/0` on amylose and cellulose.

### Phase 3

Only if periodic backbone refinement still stalls:

- add more periodic backbone DOFs
- or move some backbone freedom into the earlier network-capture stage

Those are larger changes and should not be attempted before periodic `phi` / `psi` is in place and benchmarked.
