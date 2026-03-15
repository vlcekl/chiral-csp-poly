# Staged Symmetry-Network Ordering Implementation Plan

## Goal

Improve `ordering.strategy=symmetry_network` so the search reaches the intended CSP H-bond network more reliably.

The current single-stage objective mixes:

- target-network formation,
- steric clash avoidance,
- soft/full runtime energies.

That is too conservative too early. If the optimizer must pass through transiently crowded geometries to form the `C2/C3` zipper or `C6` pitch bridge, the search can reject that path before the network is captured.

## Proposed design

Replace the current single primary `symmetry_network` DE search with a staged selector-side search:

1. `network_capture`
   - exact symmetry preserved
   - selector and anchor-aware torsions active
   - softened soft-stage repulsion
   - no full-energy term in-loop
   - weak steric gate only
   - network-family satisfaction dominates the score

2. `network_cleanup`
   - starts from the best capture candidate
   - exact symmetry preserved
   - current-strength soft/full runtime systems restored
   - existing stronger clash and energy terms restored
   - network terms still active so cleanup cannot destroy the captured motif freely

3. `backbone_refinement`
   - unchanged position in the workflow
   - remains optional
   - still follows selector-side cleanup

So the new `symmetry_network` flow becomes:

`capture -> cleanup -> optional backbone refinement`

## Why this is the least disruptive change

- The exact-symmetry coordinate construction stays intact.
- Existing H-bond metrics and literature connectivity policy stay intact.
- Existing cleanup weights stay meaningful and become stage-2 controls instead of stage-1 controls.
- The current backbone refinement stage can remain in place.
- Runtime scoring still uses reused OpenMM contexts and the existing ordering-system builder.

## Stage definitions

### Stage A: `network_capture`

Purpose:

- enter the correct H-bond connectivity basin first

Score priority:

1. weakest geometric target-network family
2. weakest like target-network family
3. geometric donor occupancy
4. geometric fraction
5. mild clash gate
6. weak soft-energy regularization

Runtime-system behavior:

- build a dedicated soft-only ordering system
- reduce soft repulsion strength relative to the cleanup stage
- do not include full-energy search terms

Recommended defaults:

- `symmetry_network_capture_enabled: true`
- `symmetry_network_capture_soft_repulsion_scale: 0.15`
- `symmetry_network_capture_min_heavy_distance_A: 1.25`
- `symmetry_network_capture_clash_penalty: 10000.0`
- `symmetry_network_capture_weight_family_min_geom: 260.0`
- `symmetry_network_capture_weight_family_min_like: 80.0`
- `symmetry_network_capture_weight_geom_occ: 180.0`
- `symmetry_network_capture_weight_like_occ: 40.0`
- `symmetry_network_capture_weight_geom_frac: 12.0`
- `symmetry_network_capture_weight_soft_energy: 0.10`
- `symmetry_network_capture_energy_clip: 3.0`

Search budget:

- reuse `symmetry_maxiter`, `symmetry_popsize`, `symmetry_polish`

Rationale:

- `family_min_geom` is the most important term because it prevents the optimizer from solving only one sub-network
- `geom_occ` remains strong because all donors should become committed early
- `soft_energy` is deliberately weak, just enough to prefer less pathological members within the same network tier
- `1.25 A` is low enough to allow approach into a forming network, but still blocks impossible atom overlap

### Stage B: `network_cleanup`

Purpose:

- clean packing and recover realistic selector arrangement after the network basin has been found

Score priority:

- the current `symmetry_network_weight_*` family
- current clash gate
- current soft/full single-point terms

Runtime-system behavior:

- use the existing ordering runtime systems and current-strength soft/full settings

Search budget:

- separate smaller DE pass around the capture winner

Recommended defaults:

- `symmetry_network_cleanup_enabled: true`
- `symmetry_network_cleanup_maxiter: 24`
- `symmetry_network_cleanup_popsize: 8`
- `symmetry_network_cleanup_polish: false`
- `symmetry_network_cleanup_init_jitter_deg: 6.0`

Rationale:

- capture should do the hard basin finding
- cleanup only needs to refine locally around the captured network

### Stage C: `backbone_refinement`

No conceptual change:

- still optional
- still selector/backbone coupled
- now starts from the selector-side cleanup winner instead of the capture winner

## Implementation outline

### 1. Add a stage-specific objective descriptor

Introduce an internal dataclass such as `_NetworkObjectiveStage` carrying:

- clash gate
- clash penalty
- network weights
- energy weights
- energy clip
- whether full-energy search is allowed

This removes hard-coded dependence on a single global score layout and lets capture and cleanup reuse the same evaluator.

### 2. Add a capture-spec builder

Create a helper that derives a soft-only capture `OrderingSpec` from the current one:

- `skip_full_stage=true`
- `soft_repulsion_k_kj_per_mol_nm2 *= symmetry_network_capture_soft_repulsion_scale`

Everything else should stay as close as possible to the current soft-stage system.

### 3. Generalize candidate evaluation

Extend `_evaluate_network_candidate(...)` to accept a `_NetworkObjectiveStage`.

The same geometry/H-bond diagnostics path stays in place, but score construction becomes stage-dependent.

### 4. Add a rebased cleanup engine

Add a helper to create a new symmetry engine from the best capture coordinates while reusing the current selector-side DOFs and the stronger cleanup runtime systems.

### 5. Rewrite the primary selector-side search in `optimize_symmetry_network_ordering(...)`

New flow:

1. build original engine for cleanup-stage systems
2. build capture engine from softened soft-only spec
3. run capture DE
4. rerank capture population lexicographically
5. build cleanup engine rebased on capture winner
6. run cleanup DE
7. rerank cleanup population lexicographically
8. pass cleanup winner into existing optional backbone refinement

### 6. Add summary/report fields

Record:

- capture enabled/applied
- capture weights and clash settings
- capture search budget
- capture selected candidate metrics
- cleanup selected candidate metrics
- whether cleanup improved over capture

This is important because current failures are difficult to diagnose without knowing whether the network was ever captured.

## Tests

Add focused regression coverage for:

1. staged search metadata is written into the ordering summary
2. capture stage uses softened repulsion / soft-only search settings
3. cleanup stage starts from the capture winner
4. backbone refinement still runs after cleanup
5. disabling capture restores the old single-stage behavior

## Acceptance criteria

The implementation is successful if:

- `symmetry_network` still preserves exact selector screw symmetry
- the staged branch remains opt-in only through `ordering.strategy=symmetry_network`
- the default single-start behavior becomes more likely to find nonzero target-network occupancy
- cleanup no longer destroys the network captured in stage A
- summaries clearly expose which stage succeeded or failed

## Known limits

This plan does not solve:

- glycosidic `psi`
- periodic `bb_phi`
- dynamic pair reassignment beyond the targeted CSP connectivity graph

Those remain separate follow-on tasks.
