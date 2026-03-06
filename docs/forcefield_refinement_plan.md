# Forcefield Refinement Plan

## Purpose

This plan replaces the recent heavy-atom pre-relaxation focus with a plan that returns the project to the high-level targets in `scratch/updated_plan.md`.

The central correction is:

- the current generic-bonded, soft-repulsion OpenMM path is not the desired architecture,
- ordering and structural optimization must run on a real all-atom GLYCAM/GAFF forcefield,
- the intended optimization protocol is two-stage:
  - stage 1 uses realistic bonded terms with soft-repulsion nonbonded interactions to resolve overlaps,
  - stage 2 switches to full realistic nonbonded interactions for final refinement,
- hydrogens must therefore become first-class atoms in both the structure domain and the forcefield domain,
- the topology domain does not need to carry explicit hydrogens, but it must preserve enough chemistry and residue-state metadata to add them later without ambiguity,
- AMBER-format all-atom artifacts remain a required output, but they must not be the only way the project obtains physically meaningful parameters.

This plan is therefore about finishing the actual forcefield architecture described in `scratch/updated_plan.md`, not just polishing the temporary heavy-atom fallback.

---

## Alignment With `scratch/updated_plan.md`

This plan keeps the original high-level intent intact:

1. The RDKit molecule remains the single source of truth for the assembled CSP topology and coordinates.
2. Backbone, selector, and connector are still parameterized separately.
3. The final simulation representation is still built by merging component parameters onto the master topology in OpenMM.
4. GLYCAM must stay GLYCAM, GAFF must stay GAFF, and cross-boundary interactions must be handled explicitly.
5. We continue to avoid a monolithic AmberTools run on the full complex polymer as the core parameterization strategy.

The main course correction is that we now explicitly include:

1. an all-atom forcefield-domain molecule,
2. explicit hydrogens in the structure, optimization, and ordering path,
3. real per-atom nonbonded parameters,
4. explicit 1-4 exception handling across GLYCAM/GAFF boundaries,
5. a two-stage optimization protocol that uses soft repulsion only as the stage-1 nonbonded model,
6. elimination of generic bonded fallback from any path that claims to use the forcefield.

---

## Problem Statement

The current codebase has drifted into an intermediate architecture that is useful for testing migration mechanics, but is not yet the intended molecular-modeling workflow.

Current gap:

1. `src/poly_csp/forcefield/system_builder.py` still builds a generic bonded system plus soft repulsion.
2. The current OpenMM path does not assign real backbone GLYCAM parameters.
3. The current OpenMM path does not construct a real `NonbondedForce` with all-atom charges and Lennard-Jones terms.
4. Hydrogens are present in final output and fragment parameterization, but not yet in the forcefield-domain optimization model.
5. `src/poly_csp/ordering/` still operates on geometry heuristics and heavy-atom scoring, not on the real OpenMM forcefield described in `scratch/updated_plan.md`.
6. The original two-stage optimization idea has effectively been lost during migration: soft repulsion exists, but it is tied to generic bonded models instead of real all-atom bonded terms.
7. The structure domain still builds the backbone helix from hydrogen-suppressed residue templates, which is the wrong direction for all-atom GLYCAM assignment.
8. The project risks drifting toward a late generic hydrogen-addition routine, but backbone hydrogens should instead come from explicit-H residue templates in the structure domain.

That is the wrong place to stop, because the intended workflow needs:

1. all-atom selector ordering,
2. realistic local sterics and electrostatics during selector arrangement,
3. carbamate and ester hydrogen-bond geometry represented in the actual optimization physics,
4. a real mixed GLYCAM/GAFF system in OpenMM,
5. a two-stage optimization protocol for robust overlap removal followed by full-forcefield refinement,
6. AMBER-format all-atom deliverables generated from the same chemically complete model.

---

## Target End State

The target architecture after this refinement is:

1. **Topology Domain**
   - RDKit builds the full CSP as the chemically exact master record.
   - The topology domain may remain hydrogen-suppressed for assembly and reactions.
   - The topology domain must still encode hydrogen-relevant chemistry without ambiguity:
     - linkage state,
     - terminal state,
     - selector substitution state,
     - residue identity needed to choose the correct explicit-H residue template later.

2. **Structure Domain**
   - The structure domain must be all-atom before force assignment.
   - Backbone helix construction should use explicit-H residue templates and transform all atoms, including hydrogens, by the same screw operation.
   - The structure domain should not rely on a late generic routine to guess where backbone hydrogens belong after the helix has already been built.
   - The structure-domain all-atom molecule must retain stable semantic identity metadata so it can hand off cleanly into the forcefield domain.

3. **Forcefield Domain**
   - The OpenMM `System` is fully all-atom.
   - Backbone atoms receive GLYCAM bonded and nonbonded parameters.
   - Selector atoms receive GAFF2 bonded and nonbonded parameters.
   - Connector atoms receive bonded and nonbonded parameters from the capped-monomer fragment model.
   - A real `NonbondedForce` exists.
   - 1-4 exceptions are patched explicitly according to component-aware mixing rules.
   - The forcefield domain can build two optimization-ready variants from the same parameter sources:
     - an overlap-resolution system with real bonded terms plus soft-repulsion nonbonded terms,
     - a full system with real bonded and real nonbonded interactions.

4. **Ordering Domain**
   - Selector ordering uses the real OpenMM system, not heuristic-only clash/hbond scoring and not generic bonded placeholders.
   - Ordering follows the same two-stage protocol:
     - stage 1 resolves clashes with soft repulsion,
     - stage 2 finishes on the full realistic forcefield.
   - Hydrogens participate in the forcefield used for optimization.

5. **Output Domain**
   - Final structures are all-atom.
   - The pipeline can emit AMBER-format all-atom artifacts for the finished CSP.
   - Those artifacts are downstream products, not the sole forcefield source for the optimization path.

---

## Scope Decisions

### In Scope

1. Introduce explicit-H structure-domain residue templates and all-atom backbone construction.
2. Introduce a true all-atom forcefield-domain molecule for optimization and export.
3. Replace the generic bonded OpenMM builder with real GLYCAM/GAFF/connector parameter merging.
4. Build a real all-atom `NonbondedForce`.
5. Implement explicit 1-4 scaling patching as part of the main forcefield path.
6. Reintroduce the intended two-stage optimization protocol using real bonded terms in both stages.
7. Update ordering so it evaluates and optimizes with the real OpenMM system.
8. Keep optional AMBER artifact generation, but decouple it from the internal forcefield build path.

### Out Of Scope

1. Rewriting the RDKit topology assembly to operate natively on explicit-hydrogen molecules at every stage.
2. Solvent models, PME production settings, or full MD protocol design.
3. Forcefield extensions beyond the current GLYCAM backbone + GAFF2 selector + capped-connector concept.
4. General support for arbitrary selector chemistries beyond what the current selector metadata model can represent.

---

## Guiding Design Decisions

### 1. Keep the topology domain hydrogen-suppressed, but make structure and forcefield all-atom

The project should not discard the assembled RDKit master as the central record of chemistry and coordinates.

But force assignment must no longer happen on the heavy-atom-only runtime model, and the structure domain should not postpone backbone hydrogen placement until after helix construction.

Recommended approach:

1. Keep the topology build hydrogen-suppressed if that remains the simplest way to do reactions and mapping.
2. Require the topology domain to preserve enough residue-state information that explicit hydrogens can later be introduced without ambiguity.
3. Build the structure-domain backbone from explicit-H residue templates once the chemistry is finalized.
4. Apply the same helix/screw transforms to all atoms, including hydrogens.
5. Treat the resulting all-atom structure-domain molecule as the handoff object for the real forcefield builder and ordering-time OpenMM optimization.

### 2. Stop using generic bonded terms once the forcefield is enabled

If `forcefield.options.enabled=true`, then generic bonded placeholders must not be part of the claimed forcefield path.

Recommended rule:

1. Generic bonded systems may remain only as an explicitly named fallback/debug mode.
2. Ordering, relaxation, and any “forcefield-enabled” pipeline path must fail if real GLYCAM/GAFF/connector parameter assignment is incomplete.

### 3. Keep soft repulsion only as a stage-1 nonbonded model

Soft repulsion is still useful, but only for the first stage of conformational optimization.

The correct use is:

1. stage 1:
   - real all-atom bonded terms,
   - soft-repulsion nonbonded interactions,
   - goal: remove overlaps and bad contacts safely.
2. stage 2:
   - real all-atom bonded terms,
   - real all-atom nonbonded interactions,
   - goal: finish refinement on the physically meaningful forcefield.

What is not acceptable:

1. a stage-1 system that uses generic bonded terms,
2. a “forcefield-enabled” path that never transitions to real nonbonded interactions,
3. treating soft repulsion as the production forcefield.

### 4. Treat the forcefield as component-merged but atom-explicit

The intended architecture is still modular.

That means:

1. parameterize backbone, selector, and connector separately,
2. map their parameters onto the final all-atom CSP topology,
3. explicitly resolve conflicts at the connector boundary,
4. explicitly patch cross-component 1-4 interactions.

The forcefield is not “united atom”; it is “merged from modular all-atom sources.”

### 5. Do not solve backbone hydrogens with a late generic placement routine

For the backbone, explicit hydrogens should come from explicit-H residue templates in the structure domain, not from a late generic `AddHs()`-style guess after the helix has already been built.

The right workflow is:

1. topology determines the final residue chemistry,
2. structure chooses the correct explicit-H residue template for each residue state,
3. helix construction transforms those all-atom residue templates directly,
4. forcefield assignment maps GLYCAM parameters onto that explicit-H structure.

This is required because GLYCAM needs the correct atom presence and atom naming, not just “some hydrogens somewhere.”

---

## Implementation Plan

### Phase 1: Define Explicit-H Structure-Domain Templates And The All-Atom Handoff

1. Introduce a clear API for building an explicit-H structure-domain molecule once topology is finalized.
   - Recommended module: `src/poly_csp/structure/all_atom.py`
   - Recommended APIs:
     - `select_residue_templates(topology_mol) -> list[ResidueTemplateState]`
     - `build_all_atom_backbone_structure(topology_mol, helix_spec, residue_states) -> Chem.Mol`

2. Add explicit-H residue templates for the backbone structure domain.
   - These must distinguish chemically different residue states, including:
     - internal linked residues,
     - terminal residues,
     - periodic closure variants if supported,
     - substituted residues at `O2`, `O3`, or `O6`,
     - `natural_oh` vs `anhydro` representations when relevant.
   - The structure domain must choose among these templates from topology metadata, not from an ambiguous late guess.

3. Build helix coordinates by transforming explicit-H residue templates directly.
   - The same geometric transforms must be applied to heavy atoms and hydrogens.
   - Do not add backbone hydrogens after helix construction with a generic placement routine.

4. Introduce a clear API for converting the all-atom structure-domain molecule into the forcefield-domain molecule.
   - Recommended module: `src/poly_csp/forcefield/model.py` or `src/poly_csp/forcefield/all_atom.py`
   - Recommended API:
     - `build_forcefield_molecule(mol_structure_all_atom) -> Chem.Mol`

5. The all-atom forcefield-domain molecule must:
   - include all explicit hydrogens,
   - preserve a parent-heavy mapping for each added hydrogen,
   - preserve `_poly_csp_component`,
   - preserve selector instance metadata,
   - preserve residue/site metadata,
   - preserve enough naming metadata to map into GLYCAM/GAFF fragment artifacts.

6. Define stable atom naming and indexing rules for the structure-domain and forcefield-domain molecules.
   - Backbone heavy atoms should be named from residue labels.
   - Backbone hydrogens should be named from parent residue labels plus local hydrogen serial.
   - Selector and connector atoms should be named from selector local indices plus hydrogen suffix rules.
   - Names must survive the PDB/MOL2/PRMTOP path.

7. Add a structure/forcefield-domain manifest object.
   - It should record:
     - all-atom index,
     - master heavy-atom parent index for derived H atoms,
     - component,
     - residue label,
     - selector local index if applicable,
     - connector role if applicable,
     - atom name used in fragment and export mapping.

8. Acceptance for this phase:
   - topology metadata is sufficient to choose explicit-H residue templates without ambiguity,
   - the backbone structure domain is all-atom before force assignment,
   - every atom in the all-atom structure/forcefield molecule can be mapped back to the RDKit master semantic identity,
   - all hydrogens needed for downstream force assignment exist explicitly,
   - heavy-atom coordinates are unchanged by the structure-to-forcefield handoff.

### Phase 2: Implement Real Backbone GLYCAM Parameter Loading

1. Refactor `src/poly_csp/forcefield/glycam.py` so it stops being just an export helper and becomes the source of reusable backbone parameter data.

2. Add a `GlycamParams` payload that can provide:
   - per-atom charges,
   - Lennard-Jones types/parameters,
   - bond terms,
   - angle terms,
   - proper torsions,
   - impropers,
   - source residue names and atom names.

3. Implement programmatic GLYCAM extraction from pure backbone reference systems.
   - Build pure polysaccharide reference residues or small reference oligomers with tleap.
   - Load their prmtops in OpenMM.
   - extract atom-level and bonded parameters by residue atom name.

4. Do not rely on generic covalent radii or hybridization guesses anywhere in this path.

5. Define how to map GLYCAM residue atom names to the all-atom CSP forcefield-domain molecule.
   - This mapping must be deterministic for:
     - internal residues,
     - terminal residues,
     - periodic variants if they remain supported.

5a. Runtime GLYCAM loading and AMBER export must remain separate concerns.
   - Runtime extraction/mapping/build should live in the forcefield runtime modules.
   - tleap export for downstream artifacts should stay a separate export utility.

6. Acceptance for this phase:
   - a pure polysaccharide CSP can build an all-atom OpenMM system using GLYCAM only,
   - no generic bonded fallback is used,
   - the resulting system contains a real `NonbondedForce`.

### Phase 3: Extend GAFF Selector Parameterization To Full All-Atom Transfer

1. Keep isolated selector parameterization as a separate artifact pipeline.

2. Expand the selector payload from “selector-core bonded overlays” to full all-atom selector parameter data.
   - Include:
     - per-atom charges,
     - LJ terms,
     - bonds,
     - angles,
     - torsions,
     - impropers,
     - source atom names.

3. Preserve the connector partitioning concept.
   - Selector-core terms still exclude connector-touched bonded terms.
   - But selector hydrogens and selector nonbonded parameters must be included in the selector parameter set.

4. Make `load_gaff2_selector_forces()` evolve into a more complete fragment loader, not only a bonded-term transfer utility.
   - Recommended split:
     - `load_selector_fragment_params()`
     - `transfer_selector_bonded_terms()`
     - `transfer_selector_nonbonded_terms()`

5. Acceptance for this phase:
   - isolated selector all-atom parameters can be mapped onto any attached selector instance in the full CSP,
   - selector hydrogens are present and parameterized,
   - no selector optimization path uses generic bonded fallback.

### Phase 4: Implement All-Atom Connector Parameter Extraction

1. Keep capped-monomer fragments as the connector parameter source, as intended in `scratch/updated_plan.md`.

2. Expand connector extraction so it returns:
   - all bonded terms touching the connector region,
   - per-atom charges for connector atoms,
   - Lennard-Jones parameters for connector atoms,
   - impropers if needed for carbamate/ester planarity.

3. Keep the role-based semantic mapping.
   - Backbone side by residue labels.
   - Selector side by selector local index and connector role.

4. Explicitly define connector ownership rules.
   - Any nonbonded parameter on connector atoms comes from the capped-monomer fragment.
   - Any bonded term touching one or more connector atoms comes from the capped-monomer fragment.
   - Pure selector-core terms come from the isolated selector fragment.
   - Pure backbone terms come from GLYCAM.

5. Acceptance for this phase:
   - connector atoms are all-atom and parameterized,
   - carbamate/ester hydrogens and heteroatoms are represented in the forcefield-domain system,
   - no connector boundary term is approximated by generic placeholders.

### Phase 5: Build The Real All-Atom OpenMM System

1. Redesign `src/poly_csp/forcefield/system_builder.py` around the actual target behavior from `scratch/updated_plan.md`.

2. Split the builder into explicit system-construction modes driven by the same parameter sources.
   - Recommended APIs:
     - `create_full_system(...)`
     - `create_overlap_resolution_system(...)`
   - Or a single:
     - `create_system(..., nonbonded_mode="soft"|"full")`

3. The stage-1 overlap-resolution system must:
   - accept the all-atom forcefield-domain molecule,
   - add all real bonded terms from GLYCAM/GAFF/connector sources,
   - omit realistic electrostatics and Lennard-Jones interactions,
   - replace them with soft repulsion,
   - preserve exclusions needed to avoid bonded-neighbor artifacts.

4. The full system must:
   - accept the all-atom forcefield-domain molecule,
   - initialize particles for all atoms,
   - create a real `NonbondedForce`,
   - add backbone GLYCAM parameters,
   - add selector GAFF2 parameters,
   - add connector parameters,
   - add all bonds, angles, torsions, and impropers,
   - add exclusions and explicit 1-4 exceptions,
   - apply component-aware exception scaling patches.

5. Delete or retire the generic builders from the main path:
   - `build_relaxation_system()`
   - `build_bonded_relaxation_system()`

6. If any legacy helper remains, it must be rewritten so that stage 1 still uses real bonded parameter sources, not generic bonded guesses.

7. Extend `SystemBuildResult` to return:
   - `system`,
   - `positions_nm`,
   - `nonbonded_mode`,
   - `topology_manifest`,
   - `component_counts`,
   - `exception_summary`,
   - `source_manifest`.

8. Acceptance for this phase:
   - the forcefield-enabled pipeline can create both stage-1 and stage-2 all-atom systems,
   - stage 1 uses real bonded terms plus soft repulsion,
   - stage 2 uses a real `NonbondedForce`,
   - generic bonded terms are absent from both enabled stages.

### Phase 6: Make 1-4 Exception Handling A First-Class, Tested Feature

1. Upgrade `src/poly_csp/forcefield/exceptions.py` from dormant utility to required assembly step.

2. Make exception patching operate on explicit component metadata:
   - backbone-backbone -> GLYCAM scaling,
   - selector-selector -> GAFF scaling,
   - cross-connector / cross-boundary -> configured rule, defaulting to the chemistry justified by `scratch/updated_plan.md`.

3. Ensure the exception patcher works on the all-atom system, not on the current soft-repulsion placeholder.

4. Add explicit validation:
   - every expected 1-4 pair belongs to a known component class,
   - no pair is silently left at a wrong default scale,
   - connector-boundary exceptions are reported in the build summary.

5. Acceptance for this phase:
   - exception scaling is no longer theoretical,
   - it is part of the normal `create_system()` path and covered by dedicated tests.

### Phase 7: Move Ordering Onto The Real Forcefield

1. Redesign `src/poly_csp/ordering/` so it can drive actual OpenMM minimization or restrained optimization on the all-atom forcefield-domain system.

2. Keep the existing heuristic scoring utilities only as prescreening tools if useful.

3. Introduce a forcefield-aware ordering runner with an explicit two-stage protocol, for example:
   - build the stage-1 all-atom overlap-resolution system,
   - update selector torsions in the all-atom coordinates,
   - run short minimization or restrained optimization with real bonded terms plus soft repulsion,
   - rebuild or switch to the full all-atom system,
   - run final minimization or scoring with realistic nonbonded interactions,
   - rank poses by full-forcefield objective plus optional symmetry/hbond metrics.

4. Hydrogens must be present during this process.
   - That is essential for:
     - carbamate NH orientation,
     - OH/NH donor geometry,
     - short-range sterics,
     - realistic selector-selector packing.

5. Ensure ordering can still use multistart and repeat-unit symmetry logic.

6. Acceptance for this phase:
   - `src/poly_csp/ordering/` can optimize selector configurations with the all-atom OpenMM system,
   - stage 1 uses soft repulsion only for overlap cleanup,
   - stage 2 uses the complete realistic forcefield,
   - forcefield-enabled ordering no longer relies on a heavy-atom-only surrogate.

### Phase 8: Refactor Relaxation To Use The Same All-Atom System

1. Update `src/poly_csp/forcefield/relaxation.py` so it consumes the same all-atom `SystemBuildResult` used by ordering.

2. Remove any remaining dependence on:
   - generic bonded builders,
   - split composite system assembly,
   - heavy-atom-only forcefield assumptions.

3. Keep restraints if needed, but apply them on the real all-atom system.

4. Reinstate the same two-stage optimization idea in relaxation:
   - stage 1 uses real bonded terms plus soft repulsion to remove overlap pathologies,
   - stage 2 switches to the full realistic all-atom nonbonded model for final refinement.

5. Define strict behavior:
   - if full parameterization is incomplete, forcefield-enabled relaxation fails,
   - it does not silently degrade to generic bonded terms.

### Phase 9: Make AMBER Export A Downstream Product Of The Real Model

1. Clarify the role of AMBER export.
   - It is still required as an output format.
   - It should not remain the only source of “real” parameters for internal optimization.

2. Keep the current residue-aware tleap export path as a compatibility/export route if useful, but decouple it fully from ordering and OpenMM system construction.

3. Add one of the following output strategies:

   - Preferred:
     - emit all-atom OpenMM system + coordinates as the primary internal representation,
     - use ParmEd or an equivalent bridge to write `prmtop/inpcrd` from the assembled OpenMM system.

   - Acceptable fallback:
     - retain the modular tleap export route for AMBER files,
     - but validate that the emitted AMBER system is consistent with the in-process OpenMM system on atom naming, count, charges, and topology.

4. Acceptance for this phase:
   - AMBER-format all-atom artifacts are available,
   - they are no longer entangled with the internal forcefield build path,
   - the optimization path does not depend on monolithic AMBER assembly.

---

## Configuration Changes

Add or revise forcefield options so the enabled path is explicit and not ambiguous.

Recommended settings:

- `forcefield.options.enabled=true`
- `forcefield.options.mode=all_atom_modular`
- `forcefield.options.require_full_parameters=true`
- `forcefield.options.build_nonbonded=true`
- `forcefield.options.patch_exceptions=true`
- `forcefield.options.allow_generic_fallback=false`
- `forcefield.options.forcefield_domain_hydrogens=explicit`
- `forcefield.options.ordering_uses_forcefield=true`
- `forcefield.options.optimization_protocol=two_stage`
- `forcefield.options.stage1_nonbonded=soft_repulsion`
- `forcefield.options.stage2_nonbonded=full`

Recommended rule:

- if `forcefield.options.enabled=true`, then `allow_generic_fallback` should default to `false`.

---

## Testing Plan

### Unit Tests

1. GLYCAM parameter extraction:
   - verify per-atom charge/LJ loading by atom name,
   - verify bonded terms on pure backbone residues.

2. Selector GAFF extraction:
   - verify all-atom selector parameter mapping by atom name,
   - verify hydrogen atoms are present and parameterized.

3. Connector extraction:
   - verify connector bonded and nonbonded terms are extracted from the capped-monomer fragment,
   - verify carbamate/ester planarity impropers if present.

4. Structure-domain explicit-H templates:
   - verify topology metadata selects the correct residue template state,
   - verify the built all-atom backbone contains the expected hydrogens for internal, terminal, periodic, and substituted residue states,
   - verify hydrogens are transformed with the same helix geometry as their parent residues.

5. Forcefield-domain molecule:
   - verify explicit hydrogens exist where expected,
   - verify metadata propagation from master heavy atoms,
   - verify stable atom naming and parent mapping.

6. Exception patching:
   - verify backbone-backbone, selector-selector, and cross-boundary 1-4 pairs get the expected scales.

7. Two-stage builder behavior:
   - verify stage 1 and stage 2 share the same bonded term assignment,
   - verify only the nonbonded model changes between stages.

### Integration Tests

1. Pure backbone system:
   - build an all-atom GLYCAM-only OpenMM system,
   - confirm the backbone structure handed into force assignment already includes explicit hydrogens,
   - confirm the presence of real nonbonded parameters.

2. Selector-functionalized system:
   - build a full all-atom GLYCAM/GAFF/connector system,
   - confirm hydrogens are present in the optimization model,
   - confirm no generic bonded forces are inserted.

3. Two-stage optimization path:
   - build the stage-1 overlap-resolution system and confirm soft repulsion is used instead of realistic nonbonded interactions,
   - switch to the stage-2 full system and confirm realistic nonbonded interactions are present,
   - verify atom indexing and restraints remain consistent across the stage transition.

4. Ordering:
   - run selector ordering on the all-atom forcefield-enabled path,
   - confirm stage 1 resolves clashes robustly,
   - confirm stage 2 evaluates and improves the realistic forcefield objective.

5. Relaxation:
   - run staged relaxation using the all-atom system,
   - confirm stage 1 uses soft repulsion with real bonded terms,
   - confirm stage 2 uses the full realistic forcefield,
   - confirm forcefield-enabled mode does not touch generic fallback builders.

6. AMBER output:
   - verify exported all-atom AMBER artifacts match atom count, atom naming, and total charge of the internal all-atom model.

### Regression Tests

1. Existing topology and assembly tests must continue to pass.
2. Hydrogen-handling tests must continue to pass.
3. Connector mapping tests must continue to pass.
4. Any remaining heavy-atom debug paths must be clearly marked as non-production and tested separately if retained.

---

## Acceptance Criteria

This refinement is complete when all of the following are true:

1. `forcefield.options.enabled=true` builds a real all-atom OpenMM system.
2. The topology domain can determine later hydrogen addition without ambiguity.
3. The structure domain backbone is explicit-H before force assignment.
4. The enabled system contains a true `NonbondedForce` with explicit per-atom GLYCAM/GAFF/connector parameters.
5. The enabled optimization path is two-stage:
   - stage 1 uses real bonded terms plus soft repulsion,
   - stage 2 uses the complete realistic nonbonded forcefield.
6. Hydrogens are part of the optimization model used by ordering and relaxation.
7. No generic bonded fallback is used in the enabled path.
8. Connector atoms are parameterized from capped-monomer fragments, not placeholders.
9. 1-4 exception scaling is applied explicitly and tested.
10. Ordering uses the real all-atom forcefield instead of a heavy-atom surrogate.
11. The pipeline can emit all-atom AMBER-format artifacts for the final CSP.
12. The resulting architecture remains consistent with `scratch/updated_plan.md`.

---

## Recommended Delivery Sequence

Implement in this order:

1. topology residue-state annotations that make later hydrogen addition unambiguous,
2. explicit-H structure-domain residue templates and all-atom backbone construction,
3. all-atom forcefield-domain molecule,
4. backbone GLYCAM extraction,
5. full all-atom selector transfer,
6. full all-atom connector extraction,
7. real system builder with nonbonded terms,
8. exception patching,
9. ordering migration,
10. relaxation migration,
11. AMBER-format export validation.

This order is important because ordering should not be migrated until the system builder actually produces the forcefield that ordering is supposed to optimize against.

---

## Final Note

The previous heavy-atom connector plan was not wrong as an intermediate migration step, but it is not the finish line for this project.

The finish line, per `scratch/updated_plan.md`, is:

- chemically exact RDKit assembly,
- all-atom modular forcefield assignment,
- explicit GLYCAM/GAFF/connector parameter merging in OpenMM,
- explicit nonbonded exception handling,
- all-atom forcefield-driven ordering and optimization,
- all-atom AMBER-format deliverables.

This refinement plan restores that target and should be treated as the new forcefield roadmap.
