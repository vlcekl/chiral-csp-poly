# Conformer Optimization

This document describes the conformer optimization functionality in `poly_csp`:

- selector `ordering`
- post-order `relaxation`
- the old and new ordering strategies
- how ordering and relaxation can be combined

The key architectural point is that both ordering and relaxation now operate on the same canonical all-atom runtime system family. They are not separate chemistry models.

## 1. Terminology

In this project, “conformer optimization” is split into two layers:

1. `ordering`
   - searches selector orientations on the canonical runtime molecule
   - intended to discover a good ordered CSP conformer

2. `relaxation`
   - refines a finished runtime conformer with a staged restraint policy
   - intended to remove residual strain or prepare a downstream seed

Ordering is optional. Relaxation is optional. They can be used together or independently.

## 2. The Optimization Stack

The current runtime stack is:

1. build the explicit-H structure
2. convert it into the forcefield-domain molecule
3. build the canonical runtime OpenMM systems
4. optionally run ordering
5. optionally run relaxation
6. export and QC the final conformer

The canonical runtime system is the source of truth for both ordering and relaxation:

- backbone terms come from GLYCAM
- selector terms come from GAFF-derived selector payloads
- connector terms come from capped-fragment payloads

So when conformers are compared or refined, they are compared under the intended all-atom forcefield, not under a geometry-only surrogate.

## 3. Ordering

`ordering` is selector-focused conformer optimization.

It runs before any optional post-order relaxation and writes its diagnostics into `ordering_summary` in `build_report.json`.

### 3.1 `greedy` ordering

This is the original forcefield-aware ordering path.

How it works:

1. Build a discrete selector pose library from the selector asset rotamer grid.
2. Optionally randomize the initial repeat-class assignment.
3. Sweep through sites and repeat classes greedily.
4. For each candidate pose, run a short runtime `soft -> full` minimization.
5. Keep the locally best-improving candidate and continue.

Properties:

- discrete search
- path-dependent
- repeat-aware through `ordering.repeat_residues`
- each trial is locally minimized before being scored

Objective:

- default: negative final stage-2 energy
- if `skip_full_stage=true`: negative final stage-1 energy

This is good for local basin discovery, but it does not enforce exact selector screw symmetry in the final coordinates.

### 3.2 `symmetry_coupled` ordering

This is the new symmetry-preserving branch.

How it works:

1. Identify the active selector dihedrals from the selector rotamer grid.
2. Use only the residue-0 selector degrees of freedom as independent variables.
3. For each trial:
   - apply those dihedrals on residue 0,
   - rebuild the selector and connector blocks on every other residue by exact screw projection,
   - score the resulting full-polymer coordinates by single-point energy.
4. Optimize those variables with differential evolution.

Properties:

- continuous search
- exact screw-related selector geometry during the search
- residue-0 variables only, but full-polymer energy evaluation
- no in-loop local minimization

Objective:

- default: negative full-system single-point energy
- if `skip_full_stage=true`: negative soft-system single-point energy

Important clarification:

“Single-residue optimization” here means single-residue independent variables, not single-residue energy. The scored coordinates still include all symmetry-related residues in the model, and periodic systems still include periodic-image interactions through the normal runtime OpenMM system.

### 3.3 `symmetry_network` ordering

This is the network-first symmetry branch.

How it works:

1. Start from the same residue-0 symmetry parameterization used by `symmetry_coupled`.
2. Extend the active variables with selector anchor-aware dihedrals such as `tau_attach` when the selector asset provides them.
3. Run a `network_capture` selector stage:
   - apply the residue-0 selector and anchor-aware dihedrals,
   - rebuild the selector and connector blocks on every other residue by exact screw projection,
   - compute current-geometry H-bond metrics and clash diagnostics,
   - evaluate a softened soft-stage objective where target-network formation is weighted more strongly than steric cleanup.
4. Run a `network_cleanup` selector stage from the best capture candidate:
   - restore the stronger clash and energy terms,
   - keep the network terms active,
   - refine the captured basin without letting the search drift freely back into a network-free state.
5. Optionally run a second symmetry-coupled backbone refinement pass from the best selector-side cleanup candidate.
   - This stage adds backbone torsions such as `bb_c6_omega` plus glycosidic `bb_phi` and `bb_psi` on both open-chain and periodic ordering models.
   - It can re-optimize the selector and anchor-aware torsions jointly with those backbone DOFs.
6. Optionally rerank the final population lexicographically by network quality before energy inside each DE stage.

Properties:

- continuous search
- exact screw-related selector geometry during the search
- full-polymer evaluation at every trial
- current-geometry network metrics, not static H-bond restraint pairs, drive the search
- includes anchor-aware torsions such as `tau_attach` when available
- uses a softened capture stage before the stronger cleanup stage
- can add a second exact-symmetry backbone refinement stage to recover motifs that need more than selector-side freedom

Objective:

- minimize a network-first symmetry score built from:
  - geometric donor occupancy,
  - like donor occupancy,
  - geometric H-bond fraction,
  - clash gating,
  - normalized soft/full single-point energies

This is the recommended branch when the main goal is to recover a coherent selector H-bond network under exact helical symmetry.

## 4. Relaxation

Relaxation is a separate stage from ordering.

It consumes the final runtime conformer and applies the project’s staged restraint protocol. Its diagnostics are written into `relax_summary`.

### 4.1 `runtime_relax`

This is the canonical two-stage post-order relaxation.

Stages:

1. stage 1
   - real bonded terms
   - soft-repulsion nonbonded model
   - positional / optional H-bond restraints as configured

2. stage 2
   - real bonded terms
   - full nonbonded interactions
   - optional anneal continuation

Use this when you want:

- a more physically refined final conformer
- post-order cleanup after either ordering strategy
- a vacuum/full-forcefield refinement path

### 4.2 `runtime_seed_relax`

This is the solvent-ready seed-oriented relaxation branch.

It keeps the same runtime parameterization but is configured for seed generation rather than full collapse into a vacuum-refined minimum.

Typical features:

- soft-stage finalization only
- `skip_full_stage=true`
- selector anti-stacking biasing
- bounded soft selector H-bond attraction
- annealing disabled

Use this when the goal is:

- a less collapsed conformer for later solvent cleanup or MD
- a downstream seed rather than the tightest vacuum minimum

## 5. Ordering vs Relaxation

The easiest way to think about the difference is:

- `ordering` chooses or searches selector orientations
- `relaxation` refines an already chosen conformer

Ordering answers:

- which selector torsions should the CSP use?
- should the repeat unit be optimized greedily or under exact symmetry?

Relaxation answers:

- once that conformer is chosen, how much should it be locally released and refined?

## 6. Supported Combinations

These are the useful combinations in the current pipeline.

### A. No ordering, no relaxation

Configuration:

- `ordering.enabled=false`
- `forcefield.options.relax_enabled=false`

Result:

- deterministic constructed geometry only

Use when:

- you want the raw constructed model
- you are debugging structure generation

### B. Ordering only

Configuration:

- `ordering.enabled=true`
- `forcefield/options=runtime`

Result:

- optimized selector conformer
- no separate post-order relaxation stage

Use when:

- you want ordering to define the conformer directly
- you want to compare `greedy`, `symmetry_coupled`, and `symmetry_network` without a later refinement step obscuring the difference

### C. Ordering + canonical relaxation

Configuration:

- `ordering.enabled=true`
- `forcefield/options=runtime_relax`

Result:

- selector conformer found by ordering
- then refined by the full two-stage relaxation protocol

Use when:

- you want an ordered conformer and then a more polished final structure

This is the most natural “production refinement” combination.

### D. Ordering + seed-oriented relaxation

Configuration:

- `ordering.enabled=true`
- `ordering=solvent_ready` or `ordering.skip_full_stage=true` style seed settings
- `forcefield/options=runtime_seed_relax`

Result:

- a more open, solvent-ready seed conformer

Use when:

- you care about downstream explicit-solvent cleanup
- you do not want ordering or relaxation to immediately re-collapse the selectors in vacuum

### E. Relaxation without ordering

Configuration:

- `ordering.enabled=false`
- `forcefield/options=runtime_relax` or `runtime_seed_relax`

Result:

- the deterministic constructed conformer is refined directly

Use when:

- you want only staged cleanup
- you do not want selector search at all

## 7. Choosing Between Ordering Strategies

Use `greedy` when:

- you want compatibility with the historical behavior
- you want discrete rotamer-library search
- you want each trial to include a short local minimization before scoring
- you want to sample local minima with multi-start searches

Use `symmetry_coupled` when:

- exact selector screw symmetry is important
- you want all three selector sites on a glucose unit optimized jointly
- you want the score to reflect the full polymer with symmetry-related neighbors included
- you want to avoid symmetry drift introduced by per-candidate local minimization

Use `symmetry_network` when:

- exact selector screw symmetry is still required
- selector H-bond network quality is more important than raw single-point energy alone
- you want the search to include attachment torsions such as `tau_attach`
- `symmetry_coupled` preserves symmetry but produces weak or random-looking selector networks

In practice:

- `greedy` is a local discrete search with minimization in the loop
- `symmetry_coupled` is a global continuous search with exact symmetry projection in the loop
- `symmetry_network` is a global continuous exact-symmetry search with a network-first score

## 8. Multi-Start Behavior

`multi_opt` sits outside the ordering strategy.

That means:

- `multi_opt` can wrap `greedy`
- `multi_opt` can also wrap `symmetry_coupled`
- `multi_opt` can also wrap `symmetry_network`

Each start simply runs the selected ordering strategy with a different seed and returns ranked results.

Interpretation of multi-start differs slightly:

- with `greedy`, different seeds change initialization and sweep order
- with `symmetry_coupled` and `symmetry_network`, different seeds change the differential-evolution population trajectory

## 9. What Gets Scored

### `greedy`

For each candidate:

- build candidate coordinates
- run short `soft -> full` minimization
- score the minimized result

So the score reflects a locally relaxed candidate.

### `symmetry_coupled`

For each candidate:

- build full-polymer coordinates by exact symmetry projection
- evaluate single-point energy on the prepared runtime system

So the score reflects the exact projected structure, not a post-minimization structure.

This difference matters scientifically:

- `greedy` can drift away from the intended symmetry during candidate scoring
- `symmetry_coupled` cannot, because the search coordinates are projected exactly before every score evaluation

### `symmetry_network`

For each candidate:

- build full-polymer coordinates by exact symmetry projection
- compute H-bond network metrics from the current candidate geometry
- reject or heavily penalize clashing candidates
- evaluate soft and optional full single-point energies
- combine network metrics and energies into the search score

So the score reflects the current projected structure and explicitly prioritizes network quality before energy refinement.

## 10. Practical Examples

### Greedy ordering only

```bash
python -m poly_csp.pipelines.build_csp \
  forcefield/options=runtime \
  ordering.enabled=true \
  ordering.strategy=greedy
```

### Symmetry-coupled ordering only

```bash
python -m poly_csp.pipelines.build_csp \
  forcefield/options=runtime \
  ordering.enabled=true \
  ordering.strategy=symmetry_coupled
```

### Network-first symmetry ordering only

```bash
python -m poly_csp.pipelines.build_csp \
  forcefield/options=runtime \
  ordering.enabled=true \
  ordering.strategy=symmetry_network
```

### Symmetry-coupled ordering followed by canonical relaxation

```bash
python -m poly_csp.pipelines.build_csp \
  forcefield/options=runtime_relax \
  ordering.enabled=true \
  ordering.strategy=symmetry_coupled
```

### Solvent-ready seed generation

```bash
python -m poly_csp.pipelines.build_csp \
  ordering=solvent_ready \
  forcefield/options=runtime_seed_relax \
  ordering.enabled=true
```

## 11. Output And Reporting

Key report locations:

- `ordering_summary`
- `relax_summary`
- QC fields in `build_report.json`

Useful ordering fields:

- `strategy`
- `objective`
- `final_energy_kj_mol`
- `final_score`
- `selected_pose_by_site`
- `final_selector_symmetry_rmsd_A`
- `active_anchor_dihedral_names`
- `network_score_weights`
- `final_selection_method`
- `final_hbond_family_metrics`

Useful relaxation fields:

- `protocol`
- `stage1_nonbonded_mode`
- `stage2_nonbonded_mode`
- `full_stage_skipped`
- `final_stage_nonbonded_mode`

Useful QC fields:

- `qc_screw_symmetry_rmsd_A`
- `qc_selector_screw_symmetry_rmsd_A`
- `qc_hbond_connectivity_policy`
- `qc_hbond_family_metrics`
- `qc_selector_torsion_stats_deg`
- `qc_selector_aromatic_stacking_A`

## 12. Recommended Usage

For direct comparison of ordering algorithms:

- run `ordering.strategy=greedy`, `ordering.strategy=symmetry_coupled`, and `ordering.strategy=symmetry_network`
- keep post-order relaxation off
- compare ordering summaries and QC directly

For a final refined receptor:

- choose the ordering strategy you trust
- then run `forcefield/options=runtime_relax`

For solvent-oriented downstream workflows:

- use the seed-oriented presets
- avoid immediately collapsing the structure with the full vacuum stage

## 13. Parameter Reference

This section explains the individual configuration parameters that control ordering and relaxation.

Important scope rule:

- `ordering.*` controls selector ordering
- `forcefield.options.*` controls runtime-system construction and optional post-order relaxation
- `seed_bias.*` provides shared default values for many soft-stage bias parameters used by both ordering and relaxation presets

### 13.1 Ordering gates and strategy selection

| Parameter | Scope | What it does | Matters when |
|---|---|---|---|
| `ordering.enabled` | all ordering | Turns selector ordering on or off. If `false`, the pipeline keeps the constructed selector geometry and skips the ordering stage entirely. | always |
| `ordering.strategy` | all ordering | Chooses the ordering algorithm. `greedy` uses discrete pose-library search with in-loop minimization. `symmetry_coupled` uses continuous residue-0 variables with exact screw projection and single-point scoring. `symmetry_network` uses the same projection machinery but scores candidates with a network-first objective and can activate anchor-aware torsions such as `tau_attach`. | always when ordering is enabled |
| `ordering.repeat_residues` | greedy primarily | Defines how residues are grouped into repeat classes for greedy ordering. If omitted, it defaults to the helix repeat. It does not define the number of symmetry variables in the symmetry strategies. | mainly `greedy` |
| `ordering.max_candidates` | greedy primarily | Caps the number of discrete library poses considered from the selector rotamer grid. Higher values only matter if the selector asset actually exposes more poses. | mainly `greedy` |

### 13.2 Greedy-only or greedy-dominant controls

| Parameter | What it does | Matters when |
|---|---|---|
| `ordering.max_site_sweeps` | Maximum number of greedy refinement sweeps over sites and repeat classes. Larger values allow more local refinement passes. | `ordering.strategy=greedy` |
| `ordering.randomize_initial_assignment` | Randomizes the initial repeat-class pose assignment before greedy refinement. | `greedy`, especially with seeds |
| `ordering.randomize_site_order` | Randomizes the order in which sites are visited during the greedy sweep. | `greedy` |
| `ordering.randomize_residue_order` | Randomizes the repeat-class visitation order during the greedy sweep. | `greedy` |
| `ordering.randomize_pose_order` | Randomizes the order in which pose-library candidates are tested. | `greedy` |

These randomization fields live in `OrderingSpec` and matter for reproducibility and local-minimum sampling, especially under `multi_opt`.

### 13.3 Shared symmetry-strategy differential-evolution controls

| Parameter | What it does | Matters when |
|---|---|---|
| `ordering.symmetry_maxiter` | Maximum number of differential-evolution generations. Raising it gives the global search more time but increases runtime. | `ordering.strategy=symmetry_coupled` or `symmetry_network` |
| `ordering.symmetry_popsize` | Differential-evolution population multiplier. Effective population size scales with the number of active symmetry variables. Larger values broaden search but increase cost. | `symmetry_coupled` or `symmetry_network` |
| `ordering.symmetry_polish` | Enables SciPy’s final local polish step after differential evolution. Leave this `false` if you want the pure global-search result only. | `symmetry_coupled` or `symmetry_network` |

For the current implementation:

- `symmetry_coupled` activates the selector rotamer-grid dihedral names on residue 0 for each selected site
- `symmetry_network` activates the same residue-0 selector dihedrals, adds anchor-aware dihedrals from `selector.anchor_rotamer_grid` such as `tau_attach`, runs a softened network-capture stage, then a stronger cleanup stage, and can follow with a second symmetry-coupled backbone refinement pass

For CSP carbamates over `C2/C3/C6`, that means `symmetry_network` usually searches more variables than `symmetry_coupled`.

### 13.4 Symmetry-network-only controls

| Parameter | What it does | Matters when |
|---|---|---|
| `ordering.symmetry_network_min_heavy_distance_A` | Heavy-atom clash gate used before energy terms are trusted. Candidates below this minimum distance receive a large penalty and are not treated as valid network solutions. | `ordering.strategy=symmetry_network` |
| `ordering.symmetry_network_clash_penalty` | Penalty assigned to candidates that fail the clash gate. Make this large enough that clashing candidates always rank below non-clashing ones. | `symmetry_network` |
| `ordering.symmetry_network_weight_geom_occ` | Weight on geometric donor occupancy, the strongest network-quality term in the score. Larger values make the search prioritize angle-qualified donor satisfaction more aggressively. | `symmetry_network` |
| `ordering.symmetry_network_weight_like_occ` | Weight on the looser distance-based donor occupancy term. This helps guide the search before fully geometric H-bonds are formed. | `symmetry_network` |
| `ordering.symmetry_network_weight_geom_frac` | Weight on the overall geometric H-bond fraction. This broadens the score beyond donor occupancy alone. | `symmetry_network` |
| `ordering.symmetry_network_weight_family_min_geom` | Weight on the weakest geometric target-network family. This is the main term that prevents the optimizer from winning by satisfying only one family such as `C6` while leaving the zipper unsatisfied. | `symmetry_network` with connectivity-aware policy |
| `ordering.symmetry_network_weight_family_min_like` | Weight on the weakest distance-only target-network family. This provides an earlier family-balancing signal before fully geometric H-bonds are formed. | `symmetry_network` with connectivity-aware policy |
| `ordering.symmetry_network_weight_soft_energy` | Weight on the normalized soft single-point energy term. Keep this nonzero so the search still prefers lower-strain candidates after the network terms. | `symmetry_network` |
| `ordering.symmetry_network_weight_full_energy` | Weight on the normalized full single-point energy term when full energy is included in the in-loop search. | `symmetry_network` |
| `ordering.symmetry_network_energy_clip` | Caps the normalized energy contribution so large raw energy differences do not swamp the network terms. | `symmetry_network` |
| `ordering.symmetry_network_rerank_population` | If `true`, reranks the final DE population lexicographically by network quality before energy rather than blindly taking SciPy’s best scalar objective member. | `symmetry_network` |
| `ordering.symmetry_network_use_full_energy_in_search` | If `true`, includes full-system single-point energy in each objective evaluation. Turning it off speeds the search and leaves the full energy for final reporting/reranking only. | `symmetry_network` |
| `ordering.symmetry_network_capture_enabled` | Enables the softened selector-side network-capture stage that runs before cleanup. | `symmetry_network` |
| `ordering.symmetry_network_capture_soft_repulsion_scale` | Multiplies the base soft repulsion strength used in the capture-stage runtime bundle. Lower values let the search approach the H-bond basin before full packing cleanup. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_min_heavy_distance_A` | Mild heavy-atom clash floor for the capture stage. This should stay low enough to allow approach into a forming network, but high enough to block impossible overlaps. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_clash_penalty` | Penalty assigned to capture-stage candidates that violate the mild clash floor. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_weight_geom_occ` | Capture-stage weight on geometric donor occupancy. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_weight_like_occ` | Capture-stage weight on like donor occupancy. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_weight_geom_frac` | Capture-stage weight on total geometric H-bond fraction. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_weight_family_min_geom` | Capture-stage weight on the weakest geometric target-network family. This is the dominant term that pushes the search into a balanced network basin before packing cleanup. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_weight_family_min_like` | Capture-stage weight on the weakest like target-network family. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_weight_soft_energy` | Capture-stage weight on soft single-point energy. Keep this small so capture remains network-dominated. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_capture_energy_clip` | Capture-stage cap on normalized energy contribution. | `symmetry_network` when capture is enabled |
| `ordering.symmetry_network_cleanup_enabled` | Enables the selector-side cleanup stage that starts from the capture winner and restores the stronger clash and energy model. | `symmetry_network` |
| `ordering.symmetry_network_cleanup_maxiter` | Maximum DE generations in the cleanup stage. | `symmetry_network` when cleanup is enabled |
| `ordering.symmetry_network_cleanup_popsize` | DE population multiplier in the cleanup stage. | `symmetry_network` when cleanup is enabled |
| `ordering.symmetry_network_cleanup_polish` | Enables SciPy’s local polish step for the cleanup stage. | `symmetry_network` when cleanup is enabled |
| `ordering.symmetry_network_cleanup_init_jitter_deg` | Jitter around the capture winner when seeding the cleanup-stage population. | `symmetry_network` when cleanup is enabled |
| `ordering.symmetry_init_from_rotamer_grid` | Seeds the DE population from the selector rotamer grid and anchor rotamer grid instead of relying only on generic Latin-hypercube initialization. | `symmetry_network` |
| `ordering.symmetry_init_jitter_deg` | Angular jitter added around rotamer-seeded initial members. Small nonzero jitter helps DE escape a purely discrete start while preserving a rotamer prior. | `symmetry_network` |
| `ordering.symmetry_backbone_refine_enabled` | Enables the second symmetry-coupled backbone refinement stage after the primary selector-side `symmetry_network` search. | `symmetry_network` |
| `ordering.symmetry_backbone_reoptimize_selectors` | If `true`, the backbone refinement stage re-optimizes the selector and anchor-aware torsions jointly with the backbone torsions instead of holding the stage-1 selector solution fixed. | `symmetry_network` when backbone refinement is enabled |
| `ordering.symmetry_backbone_include_c6_omega` | Enables the exocyclic glucose `O5-C5-C6-O6` torsion in the refinement stage. This is the main backbone-side DOF for the `C6` pitch-bridge family. | `symmetry_network` when `C6` is an active site |
| `ordering.symmetry_backbone_include_phi` | Enables the glycosidic `O5(i)-C1(i)-O4(i+1)-C4(i+1)` torsion in the refinement stage. On periodic ordering models it is applied through a lifted-chain projection step that preserves the periodic screw closure. | `symmetry_network` when `C2` and/or `C3` are active |
| `ordering.symmetry_backbone_include_psi` | Enables the glycosidic `C3(i+1)-C4(i+1)-O4(i+1)-C1(i)` torsion in the refinement stage. This complements `bb_phi` for the `C2/C3` zipper families and now works on both open-chain and periodic ordering models. | `symmetry_network` when `C2` and/or `C3` are active |
| `ordering.symmetry_backbone_maxiter` | Maximum number of differential-evolution generations in the backbone refinement stage. | `symmetry_network` when backbone refinement is enabled |
| `ordering.symmetry_backbone_popsize` | Differential-evolution population multiplier for the backbone refinement stage. Effective population size scales with the number of refinement DOFs, including selector re-optimization terms when enabled. | `symmetry_network` when backbone refinement is enabled |
| `ordering.symmetry_backbone_polish` | Enables SciPy’s local polish step for the backbone refinement stage. | `symmetry_network` when backbone refinement is enabled |
| `ordering.symmetry_backbone_init_jitter_deg` | Angular jitter around the stage-1 best candidate when seeding the backbone refinement population. This applies to backbone torsions directly and to selector/anchor torsions when they are re-optimized. | `symmetry_network` when backbone refinement is enabled |

### 13.5 Ordering restraint and protocol controls

These parameters define the runtime systems that ordering uses for scoring. Their meaning is shared across strategies, but the way they enter the score differs:

- `greedy`: candidate is locally minimized on the `soft -> full` systems
- `symmetry_coupled`: candidate is evaluated by single-point energy on the prepared systems
- `symmetry_network`: candidate is evaluated by current-geometry network metrics plus soft/full single-point terms on the prepared systems

| Parameter | What it does | Matters when |
|---|---|---|
| `ordering.positional_k` | Backbone positional-restraint strength used in the runtime ordering systems. Larger values keep the backbone closer to the constructed reference. | both strategies |
| `ordering.freeze_backbone` | If `true`, freezes the helix-core backbone heavy atoms by zeroing their masses in the runtime ordering systems. | both strategies |
| `ordering.soft_n_stages` | Number of restraint-release stages in stage 1. Only affects in-loop minimization for `greedy`; for the symmetry strategies it only affects how the shared ordering bundle is prepared, not the actual search loop. | mostly `greedy` |
| `ordering.soft_max_iterations` | Maximum iterations for each stage-1 local minimization stage. | mainly `greedy` |
| `ordering.full_max_iterations` | Maximum iterations for the stage-2 full-forcefield local minimization. | mainly `greedy` |
| `ordering.final_restraint_factor` | Fraction of the initial restraint strength retained at the end of the stage-1 release schedule. | mainly `greedy` |
| `ordering.skip_full_stage` | If `true`, ordering ends on the soft system and never evaluates the full-stage objective. For `greedy`, this means no stage-2 minimization. For `symmetry_coupled`, this means the final reported energy stays on the soft single-point system. For `symmetry_network`, it also disables full-energy contributions and final reporting falls back to the soft single-point system. | all strategies |

### 13.6 Shared soft-stage bias and nonbonded controls

These fields shape the soft-stage physics and are shared conceptually between ordering and relaxation. In the YAML presets their defaults often come from `seed_bias.*`.

| Parameter | What it does | Matters when |
|---|---|---|
| `ordering.soft_repulsion_k_kj_per_mol_nm2` | Strength of the soft repulsion used in the soft nonbonded model. Larger values penalize close overlaps more strongly. | both strategies; most visible in soft-stage or soft-only modes |
| `ordering.soft_repulsion_cutoff_nm` | Cutoff radius of the soft repulsion. Larger values make the overlap penalty act at longer range. | both strategies |
| `ordering.anti_stacking_sigma_scale` | Inflates selector aromatic LJ contact size in the soft model to discourage aromatic collapse or stacking during seed generation. | both strategies; especially seed-oriented workflows |
| `ordering.soft_exclude_14` | If `true`, excludes 1-4 interactions in the soft model. This can help keep seed structures more open. | both strategies; mainly solvent-ready/seed modes |
| `ordering.hbond_k` | Adds explicit H-bond distance restraints to the ordering systems. Stronger values bias the search toward the selected H-bond pattern. This is still most meaningful for `greedy` and `symmetry_coupled`; `symmetry_network` does not rely on these static restraints as its primary network objective. | all strategies when nonzero |
| `ordering.ideal_hbond_target_nm` | Optional target distance for explicit H-bond restraints. If unset, the pair builder uses its own default geometry target. | both strategies when `hbond_k > 0` |
| `ordering.hbond_neighbor_window` | Restricts candidate H-bond partners by residue-neighbor distance along the chain. | both strategies when H-bond restraints or soft H-bond bias are active |
| `ordering.hbond_pairing_mode` | Selects how donor/acceptor pairs are constructed. `legacy_all_pairs` is broad; `nearest_unique` is more selective. | both strategies when H-bond restraints are active |
| `ordering.hbond_restraint_atom_mode` | Chooses whether restraints are built using hydrogens when present or donor heavy atoms only. | both strategies when H-bond restraints are active |
| `ordering.hbond_max_distance_A` | Geometric cutoff used for H-bond diagnostic detection in ordering summaries. It does not directly change the force unless explicit H-bond restraints are also enabled. | both strategies, mainly diagnostics |
| `ordering.hbond_connectivity_policy` | Chooses whether H-bond metrics use the generic nearby-pair definition or a connectivity-aware CSP policy. `custom_v1` is the default and targets `C3(i)-NH -> C2(i+1)=O` plus `C6(i)-NH -> C6(i+1)=O` when those sites are active; `auto` currently resolves to that same targeted graph when it is supported and otherwise falls back to `generic`. `csp_literature_v1` remains available as the older zipper-plus-pitch policy. | diagnostics for all strategies; primary search term for `symmetry_network` |
| `ordering.hbond_min_donor_angle_deg` | Donor-angle threshold for geometric H-bond diagnostics. | diagnostics |
| `ordering.hbond_min_acceptor_angle_deg` | Acceptor-angle threshold for geometric H-bond diagnostics. | diagnostics |

### 13.7 `ordering.soft_selector_hbond_bias.*`

This nested block adds the bounded soft-stage selector carbamate H...O attraction used by the solvent-ready presets.

| Parameter | What it does | Matters when |
|---|---|---|
| `ordering.soft_selector_hbond_bias.enabled` | Enables the bounded soft selector H-bond attraction in the soft-stage model. | both strategies in soft-stage/seed workflows |
| `ordering.soft_selector_hbond_bias.epsilon_kj_per_mol` | Depth of the attractive well. Larger values reward matching selector H-bond contacts more strongly. | when enabled |
| `ordering.soft_selector_hbond_bias.r0_nm` | Center of the attractive window. | when enabled |
| `ordering.soft_selector_hbond_bias.half_width_nm` | Half-width of the attractive window around `r0_nm`. A narrow value makes the attraction more geometry-specific. | when enabled |
| `ordering.soft_selector_hbond_bias.hbond_neighbor_window` | Residue-neighbor window used when selecting which selector H-bond pairs get this soft-stage attraction. | when enabled |

### 13.8 Runtime and relaxation gates

These are the top-level switches for the post-order refinement stage.

| Parameter | What it does | Matters when |
|---|---|---|
| `forcefield.options.enabled` | Builds the canonical runtime OpenMM system. Ordering requires this to be `true`. | always for ordering or relaxation |
| `forcefield.options.relax_enabled` | Turns the separate post-order relaxation stage on or off. The `runtime_relax` and `runtime_seed_relax` presets set this for you. | relaxation only |
| `forcefield.options.cache_enabled` | Enables runtime-parameter caching for GLYCAM, selector, and connector payloads. | performance and reproducibility |
| `forcefield.options.cache_dir` | Optional cache location override. | performance and reproducibility |

### 13.9 Relaxation restraint and protocol controls

These parameters define the actual behavior of `runtime_relax` and `runtime_seed_relax`.

| Parameter | What it does | Matters when |
|---|---|---|
| `forcefield.options.positional_k` | Backbone positional-restraint strength during relaxation. Larger values hold the reference backbone geometry more tightly. | `relax_enabled=true` |
| `forcefield.options.dihedral_k` | Optional selector dihedral-restraint strength during relaxation. This is a relaxation-only control; ordering does not currently use a nonzero selector dihedral restraint. | relaxation |
| `forcefield.options.hbond_k` | Explicit H-bond distance-restraint strength during relaxation. | relaxation when nonzero |
| `forcefield.options.freeze_backbone` | Freezes helix-core backbone heavy atoms in the relaxation systems. | relaxation |
| `forcefield.options.soft_n_stages` | Number of stage-1 restraint-release stages in relaxation. | relaxation |
| `forcefield.options.soft_max_iterations` | Maximum local-minimization iterations for each stage-1 release step. | relaxation |
| `forcefield.options.full_max_iterations` | Maximum local-minimization iterations in stage 2. Also reused after annealing for the final cleanup minimization. | relaxation |
| `forcefield.options.final_restraint_factor` | Final restraint fraction retained at the end of the stage-1 release schedule. | relaxation |
| `forcefield.options.skip_full_stage` | If `true`, relaxation stops after the soft stage. This is the main switch that turns a workflow into a soft-only seed finalization path. | relaxation |

### 13.10 Shared soft-stage controls reused by relaxation

The following `forcefield.options.*` fields have the same physical meaning as their `ordering.*` counterparts, but they apply to the separate relaxation stage instead of ordering:

- `soft_repulsion_k_kj_per_mol_nm2`
- `soft_repulsion_cutoff_nm`
- `anti_stacking_sigma_scale`
- `soft_exclude_14`
- `ideal_hbond_target_nm`
- `hbond_neighbor_window`
- `hbond_pairing_mode`
- `hbond_restraint_atom_mode`
- `soft_selector_hbond_bias.*`

Use them the same way:

- stronger soft repulsion / larger sigma scaling -> more open soft-stage seeds
- `skip_full_stage=true` -> preserve that more open seed
- lower or zero soft-stage biasing -> less intervention, more forcefield-native cleanup

### 13.11 `forcefield.options.anneal.*`

These only matter when the relaxation stage actually reaches the full system.

| Parameter | What it does | Matters when |
|---|---|---|
| `forcefield.options.anneal.enabled` | Enables optional temperature ramp / heat-cool continuation on the full-stage context after the two-stage minimization. | relaxation with full stage |
| `forcefield.options.anneal.t_start_K` | Starting temperature of the anneal schedule. | when anneal is enabled |
| `forcefield.options.anneal.t_end_K` | Peak or end temperature of the anneal schedule. | when anneal is enabled |
| `forcefield.options.anneal.n_steps` | Number of MD integration steps used for the anneal schedule. | when anneal is enabled |
| `forcefield.options.anneal.cool_down` | If `true`, runs a heat/cool cycle. If `false`, runs a one-way temperature ramp. | when anneal is enabled |

Important compatibility rule:

- `anneal.enabled=true` is incompatible with `skip_full_stage=true`

because annealing is applied only on the full-stage context.

### 13.12 `seed_bias.*`

`seed_bias.*` is not a separate optimization stage. It is a shared source of default values for the soft-stage bias parameters used by both ordering and relaxation presets.

Use it when you want:

- one consistent solvent-ready bias package across ordering and relaxation
- stage-local overrides to remain available without duplicating full YAML blocks

If you override a field directly under `ordering.*` or `forcefield.options.*`, that stage-local value wins over the `seed_bias.*` default.

## 14. Summary

The current conformer optimization model is:

- `ordering` chooses the selector conformer
- `relaxation` refines that conformer
- `greedy` is the original discrete minimized-in-loop search
- `symmetry_coupled` is the new continuous exact-symmetry search
- `symmetry_network` is the network-first exact-symmetry search with anchor-aware torsions

They are complementary stages, not competing forcefields.
