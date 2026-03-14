# Network-First Symmetry Ordering Implementation Plan

## 1. Goal

Add a new symmetry-preserving ordering branch that is explicitly optimized for selector H-bond network quality, not just low soft-stage energy.

This plan also adds the missing attachment-bond torsion degree of freedom so the optimizer can rotate the whole carbamate relative to the glucose oxygen.

Recommended rollout:

- keep the current `ordering.strategy=symmetry_coupled` branch unchanged as the energy-only / seed-oriented symmetry path
- add a new `ordering.strategy=symmetry_network` branch for CSP conformer construction
- benchmark both before deciding whether `symmetry_network` should replace `symmetry_coupled` as the default symmetry mode

That avoids breaking the existing seed-generation use case while giving us a dedicated branch for network-focused conformer generation.

## 2. What The Recent Runs Show

The current symmetry branch preserves screw symmetry, but it is not selecting good carbamate networks.

Observed behavior from the recent periodic amylose runs:

- selector symmetry RMSD is effectively zero, so the branch is enforcing symmetry correctly
- min heavy-atom distance improves strongly, so the search is doing useful clash relief
- H-bond-like fraction stays flat and geometric donor occupancy only improves slightly
- multiple different residue-0 torsion vectors produce nearly identical H-bond QC

The code explains why:

1. `symmetry_coupled` searches on soft single-point energy in the inner loop.
2. The optional `ordering.hbond_k` restraint is static:
   - pairs are selected once from the starting geometry
   - if `ideal_hbond_target_nm` is unset, each target distance defaults to the measured starting distance
   - the restraint is distance-only and harmonic
3. The search space is missing the attachment-bond torsion, so the optimizer cannot swing the carbamate head into new network geometries.
4. The symmetry branch currently uses the rotamer grid only to identify active torsion names, not to preserve any rotamer prior in the search.

Conclusion:

- `ordering.hbond_k` is not the right mechanism for network discovery in the symmetry branch
- the network objective must be computed from the current candidate geometry on every evaluation
- the attachment torsion must become an active optimization variable

## 3. Design Principles

The implementation should satisfy these constraints:

- preserve exact screw symmetry throughout the search
- reuse the current runtime OpenMM system preparation and context reuse
- avoid per-candidate OpenMM system rebuilds
- avoid dynamic H-bond restraint reconstruction inside the OpenMM inner loop
- remain compatible with existing selector assets and greedy ordering where possible
- make the new network-first behavior opt-in at first

Important non-goals for the first version:

- no new periodic proxy-cell optimization model
- no full in-loop unconstrained minimization for every DE candidate
- no attempt to solve arbitrary multi-state selector pairing beyond the current near-neighbor CSP use case

## 4. High-Level Design

### 4.1 New strategy

Add:

- `ordering.strategy = "symmetry_network"`

Keep:

- `ordering.strategy = "greedy"`
- `ordering.strategy = "symmetry_coupled"`

Intent:

- `greedy`: legacy discrete minimized-in-loop ordering
- `symmetry_coupled`: current exact-symmetry, energy-only branch; still useful for seed-oriented workflows
- `symmetry_network`: new exact-symmetry, network-first branch for CSP conformer construction

This is the least disruptive way to introduce the new behavior. It avoids silently changing the semantics of `symmetry_coupled`, which is already implemented and documented.

### 4.2 Core algorithm

`symmetry_network` should reuse the current symmetry engine structure:

1. optimize only residue-0 selector torsions
2. exact-project all other selector blocks by the helix screw transform
3. evaluate the resulting full-polymer coordinates with reused OpenMM contexts
4. compute network metrics directly from the candidate geometry on every evaluation
5. optimize a network-first scalar score, not soft energy alone

The branch should still return exact-symmetry final coordinates by construction.

## 5. Why Not Use Dynamic OpenMM H-Bond Restraints

It is tempting to extend `ordering.hbond_k`, but that is not the right first implementation for the symmetry branch.

Reasons:

1. The current pair list is built from one starting geometry, not from the current candidate.
2. Dynamic pair reassignment would require rebuilding forces or systems inside the DE loop, which is too expensive and invasive.
3. The current restraint is distance-only. It does not reward donor or acceptor angles, which is exactly where the poor conformers are failing.
4. The harmonic energy scale is too weak and too local to serve as the primary search signal in a high-strain soft-energy landscape.

Recommendation:

- keep `ordering.hbond_k` as a separate feature for greedy ordering and later polishing
- do not rely on it for the new symmetry-network search objective

## 6. Objective Function Design

## 6.1 What the objective should optimize

The primary optimization target should be directional network organization, not pair-count fraction over all donor/acceptor combinations.

The best current metrics already exist in `compute_hbond_metrics()`:

- `geometric_donor_occupancy_fraction`
- `like_donor_occupancy_fraction`
- `geometric_fraction`
- `like_fraction`

For this use case:

- donor occupancy is the correct primary target
- pair fractions are secondary diagnostics

Reason:

- pair fractions scale with the total number of possible donor/acceptor contacts and become diluted in dense systems
- donor occupancy answers the physically relevant question: how many carbamate donors participate in the intended network

## 6.2 Proposed score structure

Use a network-first score of the form:

```text
score(candidate) =
    clash_penalty(candidate)
  + w_geom_occ * (1 - geometric_donor_occupancy_fraction)
  + w_like_occ * (1 - like_donor_occupancy_fraction)
  + w_geom_frac * (1 - geometric_fraction)
  + w_soft * soft_energy_term
  + w_full * full_energy_term
```

Lower is better.

Recommended behavior:

- `geometric_donor_occupancy_fraction` is the dominant term
- `like_donor_occupancy_fraction` is secondary
- `geometric_fraction` is tertiary
- soft/full energy terms are tie-breakers once the network is acceptable

## 6.3 Clash handling

The energy terms should not be allowed to dominate the early search because the current baseline exact-symmetry projection can begin with severe selector clashes.

Use an explicit clash gate:

```text
if min_heavy_distance_A < symmetry_network_min_heavy_distance_A:
    return large_penalty + clash_weight * (cutoff - dmin)^2
```

Implementation detail:

- compute `dmin` with the existing fast minimum-distance utilities from `ordering.scoring`
- skip full-energy evaluation for obviously clashing candidates

Recommended initial cutoff:

- around `1.6-1.8 A` for heavy-atom minimum distance in the search objective

This keeps the optimizer from wasting effort on full-energy evaluations for impossible geometries.

## 6.4 Energy normalization

The energy terms must be normalized or capped; otherwise they will swamp the network terms.

Recommended approach:

- compute baseline soft and full single-point energies once
- convert candidate energies to relative terms:

```text
soft_energy_term = clip((E_soft - E_soft_baseline) / max(abs(E_soft_baseline), E_floor), lo, hi)
full_energy_term = clip((E_full - E_full_baseline) / max(abs(E_full_baseline), E_floor), lo, hi)
```

This makes the objective weights interpretable and avoids giant absolute-energy swings dominating the network score.

## 6.5 Final candidate selection

Do not trust a single scalar objective alone for the final winner.

After DE finishes:

1. collect the final DE population and the baseline candidate
2. evaluate each candidate with the full metric set
3. choose the winner by lexicographic ranking:
   - maximize `geometric_donor_occupancy_fraction`
   - maximize `like_donor_occupancy_fraction`
   - maximize `min_heavy_distance_A`
   - minimize `full_energy_kj_mol`
   - minimize `soft_energy_kj_mol`

This provides a stable final choice even if the scalarized DE objective is only approximate.

Using final-population reranking is low-disruption because SciPy DE already returns the population when `polish=False`.

## 7. Attachment-Bond Torsion DOF

## 7.1 The missing variable

For carbamate selectors, the missing DOF is the torsion around the sugar-oxygen to carbonyl-carbon bond.

Conceptually:

- `tau_attach = (site_carbon, site_oxygen, carbonyl_c, amide_n)`

This is the torsion that lets the whole carbamate unit swing relative to the glucose residue.

Without it, the search can only rotate the downstream selector body while the carbamate head remains effectively fixed.

## 7.2 Minimal schema change

Do not overload the existing `SelectorTemplate.dihedrals` field with mixed integer and symbolic references. That would ripple through greedy ordering, torsion statistics, and the selector asset loader.

Instead add a parallel field for anchor-aware torsions.

Recommended asset/schema additions:

- in `SelectorAssetSpec`:
  - `anchor_dihedrals: Dict[str, tuple[str | int, str | int, int, int]] = {}`
- in `SelectorTemplate`:
  - `anchor_dihedrals: Dict[str, tuple[str, str, int, int]] = {}`

Supported symbolic atom refs for the first implementation:

- `"site_carbon"`
- `"site_oxygen"`

For carbamate assets, add:

```yaml
anchor_dihedrals:
  tau_attach: [site_carbon, site_oxygen, 2, 4]
```

where `2` and `4` are the selector-local carbonyl-carbon and amide-nitrogen map numbers already used in the asset.

## 7.3 Rotamer grid integration

`tau_attach` should be included in the selector rotamer grid so:

- greedy ordering can adopt it later if desired
- symmetry-network ordering can use the same asset-level rotamer prior

Recommended initial grid for carbamate selectors:

- four-state grid, matching the existing `tau_link`/`tau_ar` pattern
- start with `[-120, -60, 60, 120]` as a placeholder prior
- revise later if benchmark data suggests a better initial basin set

Important:

- the network-first branch should still optimize continuously
- the rotamer grid should be used as an initialization prior, not a hard discretization

## 7.4 Resolution in code

Add a shared resolver function for dihedral references:

- if the reference is an integer, resolve through the selector local-to-global map
- if the reference is `"site_oxygen"`, resolve through `site_to_oxygen_label()`
- if the reference is `"site_carbon"`, resolve through the site label itself (`C2`, `C3`, or `C6`)

This resolver should be shared by:

- `apply_selector_pose_dihedrals()`
- `symmetry_opt._active_dihedrals()`
- any future torsion-statistics extension for anchor-aware torsions

## 8. Search Initialization

The current symmetry branch uses the rotamer grid only to identify torsion names. That loses chemically meaningful prior structure.

For `symmetry_network`, use the rotamer grid values to seed the DE population.

Recommended initialization:

1. build the active torsion list, including `tau_attach`
2. for each dimension, record its discrete asset grid values
3. construct the DE initial population by sampling those values with small random jitter
4. include the current residue-0 torsion vector as one explicit population member

Benefits:

- keeps the search near chemically plausible basins
- reduces arbitrary continuous torsion drift
- remains compatible with continuous DE refinement

## 9. Candidate Evaluation Flow

For each `symmetry_network` candidate:

1. start from the base exact-symmetry coordinates
2. apply residue-0 torsion values
3. exact-project all other selector blocks
4. compute:
   - selector H-bond metrics from current coordinates
   - minimum heavy-atom distance
   - selector aromatic stacking metrics if needed
5. if clashing:
   - return penalty without expensive full-energy evaluation
6. otherwise evaluate:
   - soft single-point energy
   - full single-point energy when enabled for the strategy
7. compute the network-first scalar score
8. return both the score and cached diagnostics for possible reranking

Implementation note:

- candidate diagnostics should be cached by a rounded torsion-vector key to avoid duplicate Python-side metric work during DE revisits

This cache is optional for the first implementation but worthwhile because DE commonly re-evaluates similar points.

## 10. Proposed Configuration Surface

Add the following fields to `OrderingSpec`.

### 10.1 Strategy selection

- `strategy: Literal["greedy", "symmetry_coupled", "symmetry_network"]`

### 10.2 Network objective controls

- `symmetry_network_min_heavy_distance_A: float = 1.7`
- `symmetry_network_clash_penalty: float = 1.0e6`
- `symmetry_network_weight_geom_occ: float = 100.0`
- `symmetry_network_weight_like_occ: float = 20.0`
- `symmetry_network_weight_geom_frac: float = 5.0`
- `symmetry_network_weight_soft_energy: float = 1.0`
- `symmetry_network_weight_full_energy: float = 1.0`
- `symmetry_network_energy_clip: float = 5.0`
- `symmetry_network_rerank_population: bool = true`
- `symmetry_network_use_full_energy_in_search: bool = true`

These defaults are intentionally conservative. They must be benchmarked before becoming production defaults.

### 10.3 Optional rotamer-prior controls

- `symmetry_init_from_rotamer_grid: bool = true`
- `symmetry_init_jitter_deg: float = 10.0`

The existing `symmetry_maxiter`, `symmetry_popsize`, and `symmetry_polish` remain valid and should continue to apply to both symmetry branches.

## 11. Concrete Code Changes

## 11.1 `src/poly_csp/ordering/optimize.py`

Changes:

- extend `OrderingSpec.strategy` to include `"symmetry_network"`
- add the new network-objective config fields
- update `_ordering_objective_label()` to report the new strategy clearly
- keep the dispatcher pattern:
  - `greedy`
  - `symmetry_coupled`
  - `symmetry_network`

Reason:

- centralizes strategy selection
- avoids pipeline churn elsewhere

## 11.2 `src/poly_csp/ordering/symmetry_opt.py`

Changes:

- keep the shared engine and projection logic
- factor current energy-only logic into a reusable helper
- add a second driver:
  - `optimize_symmetry_network_ordering()`

Add helpers for:

- resolving anchor-aware torsions
- candidate metric evaluation
- network-first score construction
- DE population initialization from rotamer priors
- final-population reranking

Recommended internal structure:

- `_evaluate_symmetry_candidate(...) -> CandidateEval`
- `_network_score(...) -> float`
- `_rank_network_candidates(...) -> CandidateEval`
- `_initial_population_from_rotamers(...) -> np.ndarray | str`

Add a lightweight `CandidateEval` dataclass with:

- torsion vector
- soft energy
- full energy
- min heavy distance
- H-bond metrics
- selector stacking summary
- score

## 11.3 `src/poly_csp/ordering/hbonds.py`

Likely no required API changes for the first version.

Reuse:

- `compute_hbond_metrics()`

Possible follow-up optimization:

- add a faster lower-overhead helper that returns only donor occupancy and fractions, if profiling shows `compute_hbond_metrics()` is a bottleneck in DE

Do not first rework:

- `build_hbond_restraint_pairs()`

That function should remain the explicit-restraint helper, not become the network-search objective.

## 11.4 `src/poly_csp/topology/selectors.py`

Changes:

- add `anchor_dihedrals` to `SelectorTemplate`

Keep existing fields unchanged:

- `dihedrals`
- `rotamer_grid`
- `rotamer_max_candidates`

This preserves backward compatibility for greedy ordering and current tests.

## 11.5 `src/poly_csp/topology/selector_assets.py`

Changes:

- extend `SelectorAssetSpec` with `anchor_dihedrals`
- validate allowed symbolic refs
- resolve map-number entries for the selector-local atoms while preserving symbolic site refs
- load `tau_attach` into the template

Asset migrations:

- update carbamate selector assets:
  - `35dmpc`
  - `35dcpc`
  - `3c4mpc`
  - `3c5mpc`
  - `4c3mpc`
  - `5c2mpc`

For each:

- add `anchor_dihedrals.tau_attach`
- add `rotamer_grid.dihedral_values_deg.tau_attach`

## 11.6 `src/poly_csp/structure/alignment.py`

Changes:

- factor the current local-index dihedral resolution into a shared resolver
- teach `apply_selector_pose_dihedrals()` to apply anchor-aware torsions

Reason:

- keeps deterministic selector placement and greedy ordering compatible with the new asset field
- avoids duplicating torsion-resolution logic between alignment and symmetry ordering

## 11.7 `src/poly_csp/ordering/scoring.py`

Add or expose helpers needed by the network-first objective:

- fast min-distance calculation already exists and should be reused
- optionally add a compact helper that returns only:
  - `min_heavy_distance_A`
  - `selector_selector` min distance
  - `backbone_selector` min distance

Do not make selector torsion statistics depend on anchor-aware torsions in the first version. Those statistics are mainly QC and can stay focused on fully selector-local torsions until the network branch is stable.

## 11.8 `src/poly_csp/pipelines/build_csp.py`

Changes:

- accept `ordering.strategy=symmetry_network`
- write the new objective metadata into `ordering_summary`
- include additional summary keys:
  - network score weights
  - baseline/final donor occupancy fractions
  - baseline/final min heavy distance
  - rerank winner provenance
  - whether `tau_attach` was active

No QC schema changes are strictly required because the existing H-bond occupancy metrics already capture the relevant outcome.

## 11.9 Config files

Add:

- `conf/ordering/network_first.yaml`

It should:

- inherit from `/ordering/basic`
- switch `strategy: symmetry_network`
- set network-objective defaults explicitly
- keep `skip_full_stage: false`
- avoid solvent-ready seed bias settings by default

Do not repoint:

- `conf/ordering/basic.yaml`

until benchmarks show the new branch is clearly better.

## 12. Suggested Phase Breakdown

## Phase 1: Scoring and strategy skeleton

Deliverables:

- new `symmetry_network` strategy
- network-first objective using current H-bond metrics
- no asset changes yet

Purpose:

- isolate the value of the network-first score before touching selector schema

Success criterion:

- on the existing benchmark case, geometric donor occupancy improves relative to `symmetry_coupled`

## Phase 2: Attachment torsion

Deliverables:

- `anchor_dihedrals`
- `tau_attach` added to carbamate selector assets
- anchor-aware torsion resolution in alignment and symmetry ordering

Purpose:

- unlock the missing geometric motion once the objective is aligned

Success criterion:

- further improvement in geometric donor occupancy and visually coherent selector orientations

## Phase 3: Rotamer-prior initialization and reranking polish

Deliverables:

- DE initialization from rotamer priors
- final-population reranking
- optional energy clipping and weight tuning

Purpose:

- stabilize search quality and reduce random continuous torsion wandering

Success criterion:

- multi-start runs converge to similar QC with less seed sensitivity

## 13. Testing Plan

## 13.1 Unit tests

Add tests for:

- selector asset loading with `anchor_dihedrals`
- `apply_selector_pose_dihedrals()` applying `tau_attach`
- symmetry-network objective calling current candidate H-bond metrics rather than static pair lists
- score penalty behavior for clashing candidates
- final-population reranking selecting the expected candidate under synthetic metric inputs

## 13.2 Integration tests

Add a pipeline test that:

- runs `ordering.strategy=symmetry_network`
- confirms the branch writes the expected summary keys
- confirms selector symmetry RMSD remains near zero
- confirms the strategy reports `tau_attach` as active when using a carbamate selector

Avoid brittle tests that require a large absolute H-bond improvement on full production systems; use fixed seeds and moderate assertions.

## 13.3 Benchmarks

Create a small benchmark matrix:

- backbones:
  - amylose periodic 4/3
  - cellulose periodic 3/2 if runtime permits
- selectors:
  - `35dmpc`
  - `35dcpc`
- strategies:
  - `symmetry_coupled`
  - `symmetry_network` phase-1
  - `symmetry_network` phase-2 with `tau_attach`

Metrics:

- `qc_hbond_geometric_donor_occupancy_fraction`
- `qc_hbond_like_donor_occupancy_fraction`
- `qc_min_heavy_distance_A`
- `qc_selector_screw_symmetry_rmsd_A`
- final full single-point energy
- multi-start seed variance

The comparison should use exact same topology, selector, helix, and export settings.

## 14. Risks And Mitigations

### Risk 1: Network score overwhelms steric sanity

Mitigation:

- hard clash gate before network scoring
- keep energy and dmin as explicit tie-breakers

### Risk 2: Weight tuning becomes fragile

Mitigation:

- normalize energy terms
- use final-population lexicographic reranking
- keep weights exposed in config

### Risk 3: Anchor-dihedral schema change breaks greedy ordering

Mitigation:

- add `anchor_dihedrals` as a parallel field instead of mutating `dihedrals`
- make greedy ordering ignore the new field unless later explicitly enabled

### Risk 4: Full-energy evaluation slows DE too much

Mitigation:

- skip full energy for clashing candidates
- make full-energy-in-search configurable
- rely on reused contexts only

### Risk 5: Objective improves occupancy but produces unrealistic torsions

Mitigation:

- initialize from rotamer priors
- use reranking with full energy
- preserve existing QC checks on ring planarity, stacking, and minimum distances

## 15. Recommended First Implementation Order

If implementing now, the most effective order is:

1. add `symmetry_network` strategy and network-first objective using current H-bond metrics
2. add final-population reranking
3. benchmark against current `symmetry_coupled`
4. add `anchor_dihedrals` and `tau_attach`
5. update carbamate selector assets
6. add rotamer-prior initialization
7. retune weights and benchmark again

This sequencing isolates the two main changes:

- scoring alignment
- search-space expansion

That makes it much easier to tell which change actually fixes the poor conformers.

## 16. Bottom Line

The correct next step is not stronger static H-bond restraints. The next step is a new symmetry ordering objective that:

- scores the current candidate's network directly,
- treats geometric donor occupancy as the primary target,
- keeps symmetry exact,
- preserves steric sanity through explicit clash gating,
- and adds the missing carbamate attachment torsion.

That is the smallest change set that directly addresses the failures seen in the current runs.
