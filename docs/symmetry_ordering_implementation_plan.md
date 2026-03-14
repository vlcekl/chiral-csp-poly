# Symmetry Ordering Implementation Plan

## Goal

Add a new `ordering` strategy that optimizes selector orientations as a single symmetry-coupled problem instead of the current greedy residue/site sweep.

The new branch should:

- keep exact screw-related selector geometry throughout the search,
- optimize the three selectors on a glucose residue jointly,
- reuse the existing runtime forcefield/OpenMM machinery,
- avoid invasive changes to topology or forcefield parameterization,
- coexist with the current greedy path until it is proven better.

## What The Current Code Actually Does

After reviewing `README.md` and the ordering/runtime code path, the current behavior is:

1. `optimize_selector_ordering()` in `src/poly_csp/ordering/optimize.py` builds a pose library from the selector asset rotamer grid.
2. It applies poses with `apply_selector_pose_dihedrals()` in `src/poly_csp/structure/alignment.py`.
3. It couples assignments only at the discrete pose level by `(site, residue_index % repeat_residues)`.
4. Every candidate is then scored by a full two-stage OpenMM minimization via `_evaluate_runtime_candidate()`.
5. The minimized coordinates from the winning candidate become the new working structure for the next greedy step.

That means the current implementation has two separate symmetry leaks:

- The search is path-dependent because sites/residue classes are optimized one at a time.
- Even when the same discrete pose is assigned to symmetry-related residues, the subsequent unconstrained minimization can drift those selectors away from exact screw symmetry.

Also important: the current QC only measures backbone screw symmetry (`screw_symmetry_rmsd_from_mol()` in `src/poly_csp/ordering/scoring.py`). It does not directly verify selector symmetry.

## Critical Review Of `docs/symmetry_ordering_review.md`

The review is directionally right in a few important ways:

- A new ordering strategy is the correct integration point.
- Reusing the existing runtime bundle and OpenMM system builder is the right architectural choice.
- The optimization problem is low-dimensional if we work in the asymmetric selector degrees of freedom instead of optimizing every residue independently.
- RDKit should not be in the inner loop of a global optimizer.

But several parts of the review are too abstract or incorrect for this codebase:

### 1. The real problem is not only “greedy”

The current optimizer already couples residues through `repeat_residues`. The deeper issue is that symmetry exists only in the chosen discrete labels, not in the final minimized coordinates.

### 2. The asymmetric unit is a single glucose residue for current systems

For the supported homopolymer CSPs, the independent selector variables should be the active dihedrals on one glucose residue across the selected sites (`C2`, `C3`, `C6`), not one variable set per residue in the rational repeat.

The rational repeat is needed for box closure and reporting, but not for the number of independent selector pose parameters in a strictly screw-symmetric model.

### 3. The active variables should come from the rotamer grid, not all selector dihedrals

Example: `35dmpc` exposes `tau_link`, `tau_ar`, and `tau_ring`, but only `tau_link` and `tau_ar` are part of the current rotamer grid. A minimally disruptive symmetry optimizer should default to the same active dihedral set that the greedy optimizer already uses.

### 4. The symmetry block must include connector atoms, not only selector-core atoms

In the forcefield-domain molecule, “selector orientation” includes both selector and connector atoms. Any projection/cloning step must operate on the full attached block for each selector instance.

### 5. `apply_selector_pose_dihedrals()` is not “for isolated molecules”

It already works on attached selectors in the full polymer and is useful for setup, validation, and seed generation. It is still a bad inner-loop primitive for 10^4 objective calls, but it is not conceptually disconnected from the runtime path.

### 6. `dp * rise_A` is the unit-cell length, not the helix pitch

The current periodic box code in `src/poly_csp/structure/pbc.py` uses `Lz = dp * rise_A`. That is the axial cell length of the built polymer, not the macroscopic pitch in the general case.

### 7. “Always use a small periodic proxy system” is not the right first implementation

Building a separate repeat-cell proxy for open chains would require extra topology/runtime-parameter plumbing and creates a second optimization model to maintain. That is useful as a later refinement if end effects prove problematic, but it is not the minimally disruptive first step.

### 8. GPU-specific reasoning does not match the current runtime path

`new_context()` in `src/poly_csp/forcefield/minimization.py` explicitly prefers CPU/Reference contexts. The symmetry branch should therefore be designed around reusing a small number of CPU contexts, not around GPU context fan-out assumptions.

## Recommended Design

### Strategy

Add a second ordering strategy:

- `greedy` (existing behavior, default)
- `symmetry_coupled` (new branch)

The public entry point should remain `optimize_selector_ordering()`. Internally it becomes a dispatcher so the rest of the pipeline and `multi_opt.py` stay stable.

### Core Idea

In `symmetry_coupled` mode:

1. Optimize only the active selector dihedrals on residue 0 across the requested sites.
2. After every trial update, rebuild the full selector geometry on all residues by exact screw cloning from residue 0.
3. Score that exact-symmetry structure with reused OpenMM contexts using single-point energies, not full minimization inside the search loop.
4. Emit final coordinates by exact projection, so the output cannot drift away from symmetry.

This directly addresses the weakness of the current implementation: symmetry becomes a hard construction rule, not just an initial guess.

### Proposed Algorithm

### A. Precompute a symmetry engine

Create `src/poly_csp/ordering/symmetry_opt.py` with a private engine object, for example `SymmetryOrderingEngine`, that caches:

- base coordinates from the forcefield-domain molecule,
- helix screw parameters,
- active sites,
- active dihedral names from the selector rotamer grid,
- residue-0 attached-block atom maps for each site,
- matching attached-block atom maps for every other residue/site,
- dihedral atom tuples and downstream masks for residue 0,
- one soft OpenMM context and one full OpenMM context created once and reused.

The attached block for a site should include all atoms carrying:

- `_poly_csp_selector_instance`
- `_poly_csp_selector_local_idx`
- matching `_poly_csp_residue_index`
- matching `_poly_csp_site`

This naturally includes connector atoms when present.

### B. Define the optimization variables

The search vector should be:

- one angle per `(site, active_dihedral_name)` on residue 0

For the common carbamate selectors over `C2/C3/C6`, that gives:

- 3 sites x 2 active dihedrals = 6 variables

This is small enough for a continuous global optimizer.

### C. Build exact-symmetry trial coordinates

For each candidate vector:

1. Copy the base coordinate array.
2. Apply the target dihedrals on residue 0 using NumPy and `set_dihedral_rad()`.
3. For each residue `i > 0`, overwrite each attached selector block by applying `ScrewTransform(theta_rad, rise_A).apply(...)` to the corresponding residue-0 block.
4. Leave the backbone coordinates unchanged.

This keeps the trial geometry exactly screw-related by construction.

### D. Score with reused OpenMM contexts

Use single-point energy evaluation only during the global search:

- search objective: soft-system potential energy
- final reporting: full-system potential energy

Recommended first implementation:

1. Build the existing runtime bundle once with `_prepare_runtime_ordering_systems()`.
2. Create one reusable context for `prepared.soft.system` and one for `prepared.full.system`.
3. For each trial:
   - `context.setPositions(...)`
   - `context.getState(getEnergy=True)`
4. Return a large penalty on OpenMM exceptions or non-finite energies.

Why soft-stage energy for the search:

- it preserves the existing seed-bias logic,
- it is smoother than raw full nonbonded energy,
- it stays within the current ordering philosophy.

Why full-stage energy for the final report:

- it keeps final ranking comparable to the current ordering summary.

### E. Use a continuous optimizer, but keep it simple

Use `scipy.optimize.differential_evolution`, seeded from the existing `seed` argument.

Recommended initial bounds:

- `[-180, 180]` degrees for each active dihedral

Recommended initial defaults:

- `maxiter`: moderate, not aggressive
- `popsize`: modest, scaled to dimension
- `polish=False` initially

There is no need to add a second internal multi-start layer. The existing `run_multi_start_optimization()` can remain the outer restart mechanism.

### F. Finalization

After the optimizer returns:

1. Rebuild the exact-symmetry coordinates from the best dihedral vector.
2. Evaluate both soft and full single-point energies on those projected coordinates.
3. Write those coordinates back to the RDKit molecule.
4. Return a summary shaped like the current ordering summary, with added symmetry-specific metadata.

## Why This Is Minimally Disruptive

This design does not require:

- a new topology representation,
- a new forcefield parameterization route,
- a new periodic proxy extraction workflow,
- per-candidate OpenMM system rebuilds,
- changes to selector asset definitions.

It reuses:

- selector asset rotamer-grid metadata,
- existing atom metadata on the forcefield-domain molecule,
- `ScrewTransform`,
- `_prepare_runtime_ordering_systems()`,
- the current `multi_opt` outer wrapper,
- the current build-report pipeline.

## Concrete Code Changes

### 1. `src/poly_csp/ordering/optimize.py`

- Add `strategy: Literal["greedy", "symmetry_coupled"] = "greedy"` to `OrderingSpec`.
- Move the current implementation into a private greedy helper.
- Keep `optimize_selector_ordering()` as the public dispatcher.

### 2. `src/poly_csp/ordering/symmetry_opt.py`

Add the new engine and driver:

- runtime preparation/context reuse
- residue/site attached-block mapping
- symmetry projection
- DE objective
- summary generation

### 3. `src/poly_csp/structure/backbone_builder.py`

Persist the helical screw parameters on the built molecule:

- `_poly_csp_helix_theta_rad`
- `_poly_csp_helix_rise_A`

The greedy path already relies on molecule-level helix metadata such as `_poly_csp_helix_repeat_residues`. Storing `theta/rise` the same way keeps the symmetry branch self-contained and avoids threading `HelixSpec` through every ordering function signature.

### 4. `src/poly_csp/ordering/scoring.py`

Add a selector symmetry metric, for example:

- `selector_screw_symmetry_rmsd_from_mol(...)`

It should compare the attached selector blocks on residue `i` and `i+1` after inverse screw mapping, analogous to the current backbone-only symmetry metric.

This metric is important because the current QC cannot directly prove that selector symmetry was preserved.

### 5. `src/poly_csp/pipelines/build_csp.py`

- Parse the new `ordering.strategy`.
- Include strategy and selector-symmetry metrics in `ordering_summary`.
- Report the new selector symmetry QC metric in the build report.
- Keep the existing greedy branch unchanged.

### 6. `src/poly_csp/ordering/multi_opt.py`

No architectural rewrite is needed.

It should continue to call `optimize_selector_ordering()`, which now dispatches by strategy. The existing outer restart logic is the correct place to handle multiple random seeds.

### 7. `conf/ordering/*.yaml`

Add `strategy: greedy` to the existing configs so the old behavior remains explicit and stable.

## Summary Shape

The new branch should preserve the current high-value keys where possible:

- `enabled`
- `objective`
- `stage1_nonbonded_mode`
- `stage2_nonbonded_mode`
- `final_energy_kj_mol`
- `final_score`
- `initial_pose_by_site`
- `selected_pose_by_site`

And add new keys such as:

- `strategy`
- `symmetry_mode: "exact_screw_projection"`
- `optimized_dihedrals_by_site`
- `soft_search_energy_kj_mol`
- `final_selector_symmetry_rmsd_A`
- `active_symmetry_dof`

For `selected_pose_by_site`, store the optimized continuous angles for the residue-0 reference site blocks. Do not pretend this branch is choosing from the discrete pose library after the fact.

## Open vs Periodic Systems

### First implementation

Use the full target molecule as-is:

- periodic target -> periodic runtime system with existing box vectors
- open target -> open runtime system with no box

This keeps the implementation small and faithful to the actual built object.

### Defer proxy-cell optimization

Only add a repeat-cell proxy mode later if benchmarks show that open-chain end effects materially bias the optimized symmetric dihedrals.

That refinement is scientifically plausible, but it is not the lowest-risk first implementation.

## Testing Plan

Add tests in three groups.

### Unit tests

1. Symmetry projection maps residue-0 attached blocks onto every residue with exact screw transforms.
2. The new engine includes connector atoms in the projected blocks.
3. The active dihedral set comes from the selector rotamer grid, not every selector dihedral.
4. The optimizer is reproducible for a fixed seed.
5. The summary reports the new strategy and symmetry metrics.

### Runtime/ordering tests

1. `symmetry_coupled` works for amylose and cellulose.
2. `symmetry_coupled` works for open and periodic runtime systems.
3. `multi_opt` still works unchanged when `OrderingSpec.strategy == "symmetry_coupled"`.

### QC/integration tests

1. Pipeline run writes `ordering_summary["strategy"] == "symmetry_coupled"`.
2. Final selector symmetry RMSD is near zero for the symmetry branch.
3. Selector torsion standard deviations across residues collapse relative to the greedy branch.
4. Existing greedy tests keep passing unchanged.

## Recommended Rollout

### Phase 1

Implement the exact-symmetry global search described above and keep the greedy strategy as the default.

### Phase 2

Benchmark on representative phases:

- `chiralpak_ad`
- `chiralpak_ib`
- `chiralcel_oz`

Compare against the greedy branch on:

- final full energy,
- H-bond metrics,
- selector symmetry RMSD,
- selector torsion standard deviation,
- runtime cost.

### Phase 3

Only after those comparisons:

- decide whether to add proxy repeat-cell optimization for open chains,
- decide whether to add an optional final symmetry-restrained polish,
- decide whether `symmetry_coupled` should become the default.

## Recommendation

Implement `symmetry_coupled` as an exact-projection, single-point-energy search over the active residue-0 selector dihedrals, using the existing runtime bundle and outer `multi_opt` wrapper.

That is the smallest change that actually fixes the central weakness of the current ordering path: selector symmetry must be enforced in the coordinates themselves, not merely suggested by the order in which discrete poses are assigned.
