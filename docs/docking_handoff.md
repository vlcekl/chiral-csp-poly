# Phase 9: Docking Handoff and AMBER Export Plan

## Executive Summary

This document outlines the strategy for Stage 9: "Make AMBER Export A Downstream Product Of The Real Model", focusing specifically on how to bridge the gap between our deterministic, highly ordered CSP polymer models and downstream enantioselectivity screening using **AutoDock Vina**.

Your central goal is to perform docking studies with diverse pairs of enantiomers against several likely conformers of the CSP. The current `poly_csp` pipeline is already positioned to deliver these conformers via the `multi_opt` multi-start optimization routine. This plan determines the precise datatypes and mechanisms to cleanly hand off this data to Vina.

---

## 1. Vina Format Requirements vs. AMBER

AutoDock Vina relies entirely on the **PDBQT** format for both the receptor (the CSP) and the ligand. 

**Shall we use the AMBER format for docking?**
No. AutoDock Vina cannot ingest AMBER topology (`prmtop`) or coordinate (`inpcrd`) files. AMBER formats encode the complete mathematically-defined forcefield (bond lengths, angles, dihedrals, 1-4 interactions), which Vina ignores because it utilizes its own grid-based empirical scoring function.

However, **AMBER export remains highly valuable** as a downstream product. Once Vina identifies a favorable, highly discriminative enantiomer binding pose, you may want to validate the stability of that pose by running explicit molecular dynamics (MD). For this, having the AMBER `prmtop`/`inpcrd` representation natively available for the receptor is essential to easily combine with the parameterized ligand.

---

## 2. Leveraging the `poly_csp` Forcefield for Better Docking

A significant pitfall in standard Vina preparation is relying on automated script utilities (like MGLTools' `prepare_receptor4.py`) to generate the PDBQT file. These tools inherently guess hydrogen placements and recalculate generic partial charges (often Gasteiger charges) from scratch.

Our `poly_csp` pipeline does the heavy lifting to combine a rigorous GLYCAM backbone with GAFF2 selectors and properly integrated connectors. We already have an all-atom OpenMM `System` equipped with precise, chemically accurate partial charges.

### Recommendation: Write Custom PDBQT with OpenMM Charges
Rather than losing our high-quality partial charge data by converting through standard generic tools, Stage 9 should include a lightweight native exporter that constructs the PDBQT file directly. It can:
1. Retain the exact explicit hydrogens placed by the `poly_csp` deterministic builder.
2. Inject the specific partial charges retrieved directly from the final OpenMM `System`.
3. Map atom representations to AutoDock atom types.

While the standard Vina scoring function is mostly oblivious to partial charges (focusing heavily on steric and hydrophobic contacts), advanced scoring functions natively supported by Vina (like AD4 scoring or specifically tuned variants) strongly incorporate electrostatics. By handing Vina our superior charges, we leave the door open for high-accuracy scoring.

---

## 3. Handling Selector Flexibility (Ensemble Docking)

CSP carbamate selectors are locally flexible, which dictates the complex chiral environment. Technically, AutoDock Vina supports defining flexible sidechains within the receptor.

However, configuring dozens of flexible selectors on a single massive polymer will astronomically inflate the torsional search space. Vina will struggle to converge, returning non-reproducible or trapped states and exponentially growing calculation times.

### Recommendation: Ensemble Docking Strategy
The solution relies completely on the strengths of the current `poly_csp` pipeline.

1. **Generation:** Use the `multi_opt` module to run your multi-start optimizations, discovering 5-10 distinct, deep, and stable local minima (conformers) of the CSP.
2. **Export:** Export each of these `multi_opt` ranks as an independent, fully rigid `receptor.pdbqt` file.
3. **Screening:** Run rigid docking of the enantiomers into *each* conformer independently. Vina excels at rigid docking speeds.
4. **Aggregation:** The true binding capability of the CSP towards a specific enantiomer is evaluated as an ensemble across these conformers (e.g., probability-weighted Boltzmann average of the scores across the conformers).

This ensures the chiral phase variations are sampled faithfully by the OpenMM forcefield, while docking runs remain fast and statistically rigorous.

---

## 4. Automating the Search Bounding Box

AutoDock Vina requires the user to define a Cartesian search space (center coordinate and specific X/Y/Z dimensions). Identifying this bounding box for a complex helical polymer manually is tedious.

### Recommendation: Automated Box Generation 
Since we construct the CSP helix progressively and deterministically, the `poly_csp` framework knows exactly where the chiral cavities lie. Stage 9 should implement a bounding box calculation that:
1. Encompasses the structural volume of the core helical segment.
2. Automatically generates a `vina_box.txt` containing the `--center_x/y/z` and `--size_x/y/z` arguments.
3. Saves this directly into the output directory alongside the receptor PDBQT files.

---

## 5. Actionable Implementation Steps for Stage 9

To fulfill both the pipeline migration (Make AMBER a downstream artifact) and prepare for docking, the architecture should be extended as follows:

1. **Decouple AMBER Export:** Move the AmberTools (`antechamber`/`tleap`) assembly completely out of the core pipeline's geometry/minimization critical path. AMBER export logic operates purely as a post-optimization "save format" that interrogates the OpenMM System and topology objects to spit out `prmtop`/`inpcrd` files.
2. **Native PDBQT Export:** Extend the `poly_csp.io` domain to generate `.pdbqt` format. Extract atom coordinates, connectivities, and GLYCAM/GAFF2 partial charges from the OpenMM structure, map AutoDock atom types, and output the PDBQT files directly into `output.export_formats`.
3. **Bounding Box Calculation Unit:** Add a small geometric utility inside the `io` or `reports` module that outputs `vina_box.txt` files describing a tight bounding box (plus a customizable buffer margin, ~5-10 Å) around the generated helix.
4. **Multi-Opt Integration:** Ensure the `multi_opt` loops emit the `.pdbqt`, `.prmtop`, and `vina_box.txt` for *every* ranked conformer output directory (`ranked_001/`, `ranked_002/`). 

### Summary
The hand-off structure should look like this:
```
outputs/ranked_001/
├── model.pdb         (Visual inspection)
├── model.prmtop      (Post-docking MD Validation)
├── model.inpcrd      (Post-docking MD Validation)
├── receptor.pdbqt    (Direct AutoDock Vina input)
└── vina_box.txt      (Vina box definition arguments)
```

Through this approach, `poly_csp` acts not only as a structural builder but as a complete screening prep ecosystem, streamlining the jump straight to AutoDock Vina.
