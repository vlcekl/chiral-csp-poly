# GLYCAM Mapping

This document records the Phase 2 pure-backbone GLYCAM contract.

## Supported Slice

Phase 2 supports only the chemically exact backbone-only slice:

- polymer: `amylose`, `cellulose`
- representation: `anhydro`
- end mode: `open`
- selectors: none
- terminal caps: none

Unsupported combinations fail before system construction. There is no automatic fallback to the generic bonded builder in this mode.

## Reference Strategy

GLYCAM parameters are not guessed from the RDKit graph.

They are extracted from complete tleap-built GLYCAM reference oligomers first, then partitioned into reusable templates:

- `dp=2` supplies the direct `terminal_reducing -> terminal_nonreducing` linkage context.
- `dp=4` supplies `terminal_reducing -> internal`, `internal -> internal`, and `internal -> terminal_nonreducing` linkage contexts.

This follows the same modeling rule used for backbone geometry: derive the chemistry from the complete valid reference first, then partition it.

## Residue Order

`poly_csp` residue order runs from the free-`C1` end toward the free-`O4` end.

For open chains, the matching GLYCAM residue-code sequence is therefore:

- amylose: `4GA ... 4GA 0GA`
- cellulose: `4GB ... 4GB 0GB`

Residue roles in `poly_csp` order are:

- `dp == 1`: `terminal_nonreducing`
- `dp >= 2`, residue `0`: `terminal_reducing`
- `dp >= 2`, middle residues: `internal`
- `dp >= 2`, last residue: `terminal_nonreducing`

## Atom Naming Boundary

The generic forcefield-domain molecule keeps Phase 1 names such as:

- heavy atoms: `C1`, `O4`, `O6`
- hydroxyl hydrogens: `HO2`, `HO3`, `HO4`, `HO6`

The GLYCAM mapping layer applies forcefield-only aliases when needed:

- `HO1 -> H1O`
- `HO2 -> H2O`
- `HO3 -> H3O`
- `HO4 -> H4O`
- `HO6 -> H6O`

This aliasing happens only inside the forcefield domain. The generic atom manifest is not rewritten into GLYCAM names.

## Runtime Modules

Phase 2 is split across these modules:

- `src/poly_csp/forcefield/glycam.py`
  Loads reusable GLYCAM templates from complete reference systems.
- `src/poly_csp/forcefield/glycam_mapping.py`
  Maps the forcefield-domain molecule onto GLYCAM residue and atom identities.
- `src/poly_csp/forcefield/system_builder.py`
  Builds the pure-backbone GLYCAM OpenMM system.
- `src/poly_csp/forcefield/amber_export.py`
  Keeps tleap export as a downstream artifact path instead of the runtime parameter source.
