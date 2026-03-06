# poly_csp Metadata Invariants

This document defines molecule-level metadata used across polymerization, attachment,
dihedral application, atom mapping, and terminal/export stages.

## Molecule Properties

- `_poly_csp_dp`:
  Degree of polymerization (residue count in the base backbone before selector addition).

- `_poly_csp_polymer`:
  Backbone polymer identity (`amylose` or `cellulose`).

- `_poly_csp_representation`:
  Monomer representation mode (`anhydro` or `natural_oh`).

- `_poly_csp_template_atom_count`:
  Atom count of the heavy-atom monomer template used by selector-placement helpers and tests.

- `_poly_csp_removed_old_indices_json`:
  JSON list of integer atom indices removed from the initial concatenated template stack.
  Used to prune backbone coordinates before conformer assignment.

- `_poly_csp_residue_label_map_json`:
  JSON list (`len == dp`) of per-residue dictionaries mapping atom labels (`C1`, `O4`,
  `O6`, etc.) to molecule-global atom indices after representation-specific edits.

- `_poly_csp_siteidx_<label>`:
  Local template site index hints (backward compatibility), e.g. `_poly_csp_siteidx_O6`.

- `_poly_csp_end_mode`:
  Terminal policy mode (`open`, `capped`, `periodic`).

- `_poly_csp_end_caps_json`:
  JSON dictionary of configured end caps.

- `_poly_csp_terminal_topology_pending`:
  Boolean indicating terminal chemistry edits are pending (`False` for implemented modes).

- `_poly_csp_terminal_meta_json`:
  JSON dictionary with terminal-mode-specific metadata (e.g., periodic closure bond indices).

- `_poly_csp_terminal_cap_indices_json`:
  JSON dictionary containing atom indices added by left/right cap operations.

- `_poly_csp_selector_count`:
  Number of selectors attached so far (monotonic with each attach operation).

- `_poly_csp_manifest_schema_version`:
  Integer schema version for the derived all-atom atom manifest / naming policy.

## Atom Properties (selector atoms)

- `_poly_csp_selector_instance`:
  Integer selector attachment instance id.

- `_poly_csp_residue_index`:
  Backbone residue index this selector is attached to.

- `_poly_csp_site`:
  Attachment site label (`C2`, `C3`, or `C6`).

- `_poly_csp_selector_local_idx`:
  Selector template-local atom index before any merge/removal operations.

- `_poly_csp_selector_name`:
  Selector template name attached to the atom instance.

- `_poly_csp_linkage_type`:
  Attachment linkage type (`carbamate`, `ester`, `ether`, ...).

- `_poly_csp_component`:
  Component tag used by the domain migration (`backbone`, `selector`, `connector`).
  Added during assembly and consumed by `topology.atom_mapping` + forcefield mixing logic.

- `_poly_csp_connector_atom`:
  Optional boolean-like marker for connector boundary atoms (reserved for capped-monomer
  parameter extraction flow).

- `_poly_csp_connector_role`:
  Semantic connector role for attached linker atoms (`carbonyl_c`, `carbonyl_o`,
  `amide_n`, etc.). Connector atoms keep selector instance metadata but must carry
  component tag `connector`.

- `_poly_csp_parent_heavy_idx`:
  Present on derived explicit-hydrogen atoms. Points to the hydrogen-suppressed
  parent heavy atom so residue/component/selector metadata can be inherited.

- `_poly_csp_residue_label`:
  Backbone residue-local atom label (`C1`, `O4`, `O6`, ...). Added on the
  derived all-atom structure/forcefield handoff for backbone atoms and inherited
  by backbone hydrogens from their parent heavy atom.

- `_poly_csp_terminal_cap_side`:
  Present on derived terminal-cap atoms (`left` or `right`) so cap hydrogens and
  exported atom names can be assigned deterministically.

- `_poly_csp_atom_name`:
  Deterministic short atom name used by the all-atom structure/forcefield handoff
  and PDB export.

- `_poly_csp_canonical_name`:
  Expanded semantic atom identity used by the all-atom manifest.

- `_poly_csp_manifest_source`:
  Indicates which handoff source class produced the atom identity
  (`backbone`, `selector`, `connector`, `terminal_cap_left`, `terminal_cap_right`).

## Invariants

1. `_poly_csp_residue_label_map_json` must always refer to valid global atom indices in the current molecule.
2. For `natural_oh` representation, `O1` may be absent on internal residues by construction.
3. Selector attachment and dihedral APIs must resolve sugar-site oxygen indices via residue label maps, not fixed block offsets.
4. Metadata must be copied forward when creating a new `Chem.Mol` from edited `RWMol` objects.
5. Component tagging must remain single-valued per atom (`backbone` xor `selector` xor `connector`).
6. Connector atoms must retain `_poly_csp_selector_instance` and `_poly_csp_selector_local_idx` so they can be remapped from capped-fragment parameters back onto the full polymer.
7. Explicit H atoms inherit component/residue/selector metadata from their parent heavy atom and record `_poly_csp_parent_heavy_idx`.
8. Component geometry must be derived from the complete chemically valid structure first, then pruned by removing designated atoms or hydrogens. This rule applies to glucose representation variants (`natural_oh` -> `anhydro`) and to residue-state template variants.
9. Within `build_backbone_structure()`, heavy backbone atoms are added before their hydrogens so residue label maps remain stable when later attachment hydrogens are consumed.
10. Backbone hydrogens come from explicit-H residue templates placed directly into the helix, not from a late whole-molecule generic hydrogen-addition step.
11. PDB naming should prefer `_poly_csp_atom_name` / preassigned residue info when present instead of regenerating names heuristically.
12. Forcefield-specific residue or atom identities, such as GLYCAM residue codes and hydroxyl-hydrogen aliases (`HO2 -> H2O`), must remain in the forcefield mapping layer. They must not overwrite the generic Phase 1 manifest naming policy stored on the structure/forcefield-domain molecule.
