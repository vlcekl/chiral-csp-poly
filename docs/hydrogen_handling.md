# Hydrogen Handling in the Current Pipeline

This document describes the current Phase 1 hydrogen model.

The boundary is now explicit:

- The topology domain is hydrogen-suppressed.
- The structure domain is the first canonical all-atom representation.
- The forcefield-domain handoff validates and names that all-atom structure; it does not rebuild it.
- Backbone hydrogens never come from a late whole-molecule `AddHs()` step.

## Core Rules

1. Topology preserves chemistry and residue state, not explicit backbone hydrogens.
2. Structure builds the full all-atom backbone directly from explicit-H residue templates.
3. If a representation or residue state removes atoms, geometry must be derived from the complete chemically valid structure first and pruned afterward.
4. `complete_with_hydrogens()` is no longer part of backbone construction. It remains only for isolated fragment-preparation utilities.

## Stage 1: Glucose monomer templates

`make_glucose_template()` in [src/poly_csp/topology/monomers.py](/home/lukas/work/projects/poly_csp/src/poly_csp/topology/monomers.py) is still topology-domain code, so the stored template is heavy-atom only.

Important details:

1. The mapped SMILES are authored as complete `natural_oh` glucose monomers for amylose and cellulose.
2. Hydroxyl oxygens are normalized back to implicit-hydrogen behavior after SMILES parsing, so later bond formation can consume those hydrogens through valence and sanitization.
3. Embedding is performed on a temporary explicit-H copy.
4. `anhydro` is not embedded separately. The code embeds the full `natural_oh` structure first and then removes `O1`.

That last point is deliberate. The anhydro geometry is derived from the complete molecular structure before removing the designated atom, so `C1` keeps the correct tetrahedral geometry.

## Stage 2: Heavy-atom topology assembly

Topology assembly is still heavy-atom only:

1. `polymerize()` builds the `O4(i)-C1(i+1)` heavy-atom graph.
2. `apply_terminal_mode()` records terminal intent and periodic topology edits.
3. `resolve_residue_template_states()` determines, residue by residue, which atoms and hydrogens must exist in the later explicit-H structure.

At this stage:

- free hydroxyls are still implicit on heavy atoms,
- internal `natural_oh` residues may have `O1` removed,
- selector substitution state is tracked from topology metadata and graph connectivity.

## Stage 3: Explicit-H backbone templates

Backbone hydrogens are introduced in the structure domain through [src/poly_csp/structure/templates.py](/home/lukas/work/projects/poly_csp/src/poly_csp/structure/templates.py).

The flow is:

1. `load_explicit_backbone_template()` starts from the deterministic heavy-atom glucose template and adds explicit hydrogens.
2. Hydrogen coordinates are optimized with heavy atoms fixed.
3. `build_residue_variant()` removes only the atoms or hydrogens that the resolved residue state says should be absent.

Examples:

- missing `O1` for internal `natural_oh` residues,
- missing `HO4` for outgoing glycosidic bonds,
- missing `HO2` / `HO3` / `HO6` for substituted sites,
- missing `C1` hydrogen when a left cap occupies that site.

The geometry rule is the same as for monomers: build the full chemically complete residue first, then prune. The code never embeds a partially deleted residue and treats that geometry as authoritative.

## Stage 4: Direct all-atom backbone construction

`build_backbone_structure()` in [src/poly_csp/structure/backbone_builder.py](/home/lukas/work/projects/poly_csp/src/poly_csp/structure/backbone_builder.py) is now the canonical backbone builder.

It does three things:

1. fit one residue-local backbone pose that is compatible with the requested helix screw transform,
2. apply that same transform to every atom in each residue variant,
3. assemble the full explicit-H backbone and terminal caps in the structure domain.

Current local linkage constraints are:

- `O4-C1` bond length,
- `C4-O4-C1` donor angle,
- `O4-C1-O5` acceptor angle,
- `O4-C1-C2` acceptor-side stereochemistry angle.

The builder also exposes per-linkage diagnostics for:

- the same covalent geometry terms,
- `O4···H1` separation, which protects against the recent anomeric-placement regression.

There is no heavy-backbone retrofit step and no late generic backbone hydrogen placement.

## Stage 5: Selector attachment

Selector definitions now live as packaged selector assets under `src/poly_csp/assets/selectors/`, and they are loaded into `SelectorTemplate` objects through the shared selector registry.

When `attach_selector()` runs:

1. it starts from the all-atom backbone structure,
2. it consumes the sugar attachment hydrogen explicitly if present,
3. it bonds in the selector fragment,
4. it propagates selector and connector metadata onto attached hydrogens.

So selector attachment is no longer a heavy-atom-only chemistry edit followed by late hydrogen completion.

## Stage 6: Forcefield-domain handoff

`build_forcefield_molecule()` in [src/poly_csp/forcefield/model.py](/home/lukas/work/projects/poly_csp/src/poly_csp/forcefield/model.py) performs the structure-to-forcefield handoff.

It validates that:

1. all hydrogens are explicit,
2. hydrogens record `_poly_csp_parent_heavy_idx`,
3. backbone atoms and hydrogens carry the metadata needed for naming and later parameter mapping.

It then builds the deterministic atom manifest and export naming. It must not alter chemistry or coordinates.

## Where Generic Hydrogen Completion Still Exists

`complete_with_hydrogens()` in [src/poly_csp/structure/hydrogens.py](/home/lukas/work/projects/poly_csp/src/poly_csp/structure/hydrogens.py) still exists, but it is no longer part of canonical backbone construction.

Its remaining use is isolated fragment preparation, for example:

- AmberTools / GAFF fragment setup in [src/poly_csp/forcefield/gaff.py](/home/lukas/work/projects/poly_csp/src/poly_csp/forcefield/gaff.py),
- standalone hydrogen-completion utilities and tests.

That distinction matters:

- backbone and final structure construction are template-driven,
- generic hydrogen completion is now only a fragment utility.

## Summary

The current hydrogen model is:

1. heavy-atom topology for assembly and residue-state resolution,
2. direct explicit-H structure construction from complete templates,
3. validated all-atom forcefield handoff with deterministic naming,
4. no compatibility shim that rebuilds backbone hydrogens after the fact.
