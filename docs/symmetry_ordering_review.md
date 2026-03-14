# Symmetry Ordering Implementation Proposal

## 1. Executive Summary

The proposed algorithm transitions the selector ordering from a greedy, discrete, sequence-dependent rotamer sweep into a continuous, symmetry-enforced global optimization. This guarantees a perfectly periodic crystalline-like packing of the chiral selectors, aligning completely with the project's core philosophy of avoiding kinetically trapped, asymmetric disordered states.

The existing architecture within `poly_csp` is remarkably well-suited to support this alternative branch with minimal disruption. Key prerequisites—explicit-H topologies, pre-calculated helical parameters, and rigorous OpenMM state configurations—are already established.

We propose placing this implementation in a new module (`src/poly_csp/ordering/symmetry_opt.py`) and toggling it via a top-level configuration key `ordering.strategy = "symmetry_coupled"` (a natural extension to the current configuration schema).

## 2. Codebase Integration Analysis

### 2.1 The OpenMM Context and Evaluator
Currently, `ordering/optimize.py` uses `_prepare_runtime_ordering_systems` to spin up a `PreparedRuntimeOptimizationBundle`, which acts as the forcefield domain source of truth. We can completely reuse this bundle. The inner loop evaluates energy by updating positions via `context.setPositions()`, identical to how `run_prepared_runtime_optimization` operates.

### 2.2 Helical Symmetry Utilities
`structure/matrix.py` already implements `ScrewTransform(theta_rad, rise_A)` and `kabsch_align`. The necessary transformation matrix $\mathbf{H}_n$ can be generated trivially by calling `ScrewTransform.matrix(n)` and converting it into a 4x4 homogenous matrix or applying the rotation and translation sequentially to an $N \times 3$ coordinate array.
The helical parameters `theta_rad` and `rise_A` can be inferred from the existing `mol` properties (e.g., `dp`, macroscopic pitch box vector) or passed explicitly from the resolved `HelixSpec`.

### 2.3 Periodic Boundary Conditions (PBCs)
`structure/pbc.py` correctly establishes the orthorhombic box using `Lz = dp * rise_A` (which is exactly the macroscopic pitch $P$). When passing `xyz_nm` into `context.setPositions()`, OpenMM natively utilizes these box vectors, ensuring continuous H-bond network wrapping.

### 2.4 Dimensionality Reduction and Dihedrals
`structure/alignment.py` provides `apply_selector_pose_dihedrals` for isolated molecules. However, since the objective function will be evaluated by `scipy.optimize.differential_evolution` roughly $\sim 10^4$ times, modifying the RDKit conformer iteratively would be an extreme bottleneck.
**Solution:** We extract the relevant subset of rotatable atoms for Residue 0 into a NumPy-native dependency tree at initialization. Applying $\mathbf{\Phi}$ will entail purely vectorized NumPy 3D rotations, followed by `ScrewTransform` cloning.

## 3. Implementation Plan (Minimal Disruption)

### A. New Module: `symmetry_opt.py`
Create `src/poly_csp/ordering/symmetry_opt.py` containing three primary components:

#### 1. `SymmetryStateTree` (Data Structure)
A helper class that pre-computes the atomic indices of Residue 0's selectors, their rotatable bonds (the atoms defining the rotation axis), and the mask of downstream atoms to rotate.
*   **Initialization**: 
    - Identify the $D$ rotatable bonds in Residue 0. For standard carbamates, this is `tau_link` and `tau_ar` across 3 sites (C2, C3, C6), giving $D=6$.
    - Use `_downstream_mask` from `structure/alignment.py` to pre-calculate boolean arrays indicating which atom indices move when a specific bond rotates.
    - Extract the baseline $N_{res} \times N_{atoms\_per\_res} \times 3$ coordinate array.
*   **Vectorized Application**: A fast method `apply_phi(coords, phi_vector)` that takes exactly $D$ angles, uses Rodrigues' rotation formula (or equivalent quaternion math in NumPy) to rotate the specific masked atoms of Residue 0 in place, without invoking RDKit's `Chem.rdMolTransforms`.

#### 2. `evaluate_symmetric_energy(Phi)` (The Objective Function)
The closure passed to `scipy.optimize.differential_evolution`.
*   **Input**: A $D$-dimensional array $\mathbf{\Phi}$ (lengths in $[-\pi, \pi]$).
*   **Step 1: Local Mutation**: Call `SymmetryStateTree.apply_phi(base_coords, Phi)` to update Residue 0.
*   **Step 2: Helical Propagation**: Loop $n = 1 \dots N_{res}-1$.
    - For each residue $n$, apply the `ScrewTransform(theta_rad, rise_A).matrix(n)` to Residue 0's updated selector coordinates to generate perfectly symmetric clones for the rest of the unit cell. Replace the coordinates in the working $N_{total} \times 3$ array.
*   **Step 3: Evaluation**:
    - Convert coordinates to nanometers (`xyz / 10.0`).
    - Update the OpenMM context: `context.setPositions(xyz_nm)`.
    - Retrieve energy: `state = context.getState(getEnergy=True)`.
*   **Step 4: Exception Handling (Crucial Pitfall)**:
    - If OpenMM raises an exception (e.g., `openmm.OpenMMException(Particle coordinate is nan)`) due to extreme steric clash, catch it and return a massive penalty float ($10^9$ kJ/mol).
*   **Return**: `state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)`.

#### 3. `optimize_symmetry_coupled(...)` (The Entry Point)
The main driver function replacing `optimize_selector_ordering` when this strategy is selected.
*   **Setup**: Calls `_prepare_runtime_ordering_systems` to get the `PreparedRuntimeOptimizationBundle`. Creates a matching OpenMM `Context` (specifically using the `soft` parameters to leverage `soft_selector_hbond_bias`). Extracts the macroscopic pitch box vectors using `get_box_vectors_nm` and ensures the context's periodic box vectors are set.
*   **Execution**:
    - Defines bounds: `[(-np.pi, np.pi)] * D`.
    - Runs `scipy.optimize.differential_evolution(evaluate_symmetric_energy, bounds, popsize=15, maxiter=150, disp=False)`.
*   **Finalization**: 
    - Extracts `result.x` (the optimal $\mathbf{\Phi}$).
    - Runs `apply_phi` and the `ScrewTransform` one last time to generate the minimal Cartesian array.
    - Uses `update_rdkit_coords` to push these back to the `Chem.Mol`.
    - Returns the `Chem.Mol` and a summary dictionary (matching the interface of the greedy optimizer).

### B. Modifications to Existing Pipeline

1.  **`OrderingSpec` Update**: Add `strategy: Literal["greedy", "symmetry_coupled"] = "greedy"` to `OrderingSpec` in `ordering/optimize.py`.
2.  **Dispatcher Update**: Modify `build_csp.py` to check `spec.strategy`. If `symmetry_coupled`, route the call to `symmetry_opt.optimize_symmetry_coupled` instead of `optimize.optimize_selector_ordering`.
3.  **Multiprocessing**: Integrate `symmetry_opt.optimize_symmetry_coupled` into `multi_opt.py` so that we can easily run $N$ independent DE seeds concurrently.

### C. Potential Pitfalls and Verification

*   **Pitfall 1: OpenMM Context State Leakage**. 
    *   *Risk*: OpenMM Contexts hold state. If `evaluate_symmetric_energy` modifies the context unexpectedly or doesn't fully overwrite positions, energies will be wrong.
    *   *Verification*: `context.setPositions` completely overwrites the particle coordinate state each call. The topology and parameters are immutable across the run. This is safe.
*   **Pitfall 2: Coordinate Frame Mismatches**.
    *   *Risk*: The `ScrewTransform` defined in `structure.matrix` assumes the rotation axis is precisely the global $Z$-axis. If the polymer backbone is not perfectly aligned to $Z$ with the origin centered correctly, the clones will drift off-axis.
    *   *Verification*: The `poly_csp` builder rigorously aligns the initial backbone template to the $Z$-axis during `backbone_builder.py`. The translation is strictly along $+Z$. This is structurally robust.
*   **Pitfall 3: RDKit Atom Ordering Indices**.
    *   *Risk*: The NumPy mask indexing in `SymmetryStateTree` relies on a highly predictable atom ordering. If RDKit shuffles indices, the downstream masks will rotate the wrong atoms.
    *   *Verification*: `poly_csp` strictly controls atom indices during polymerization (via `_poly_csp_residue_label_map_json`). We must use `_selector_local_to_global_map` from `alignment.py` to cross-reference the correct global indices.
*   **Pitfall 4: PBC Box Vector Mismatch**.
    *   *Risk*: If the OpenMM periodic box $Z$-vector does not exactly match the macroscopic pitch ($N_{res} \times \text{rise\_A}$), covalent bonds crossing the boundary will be artificially stretched or squashed, yielding massive unphysical energies.
    *   *Verification*: `structure/pbc.py` correctly enforces `Lz = dp * rise_A`. We must ensure OpenMM's `System.setDefaultPeriodicBoxVectors` is populated with these precise vectors before spawning the `Context`.

## 4. Rationale & Advantages

- **Performance**: While $DE$ requires many function evaluations, single-point energy calculation in OpenMM with a frozen backbone and no gradient calculation is extremely fast ($\sim O(1)$ ms on GPU / CPU). Bypassing RDKit in the inner loop ensures the optimizer finishes in seconds.
- **Physical Correctness**: Solves the central problem detailed in `README.md`. It simultaneously explores interlocking inter-residue H-bonds inside the unit cell *and* across the periodic boundary, guaranteeing that the top and bottom of the unit cell match perfectly and sit in a true deep energy funnel.
- **Minimally Invasive**: Does not affect the `topology` or `forcefield` domains. It merely provides an alternative structure-domain parameter search utilizing the exact same runtime `System` OpenMM object built by `_prepare_runtime_ordering_systems`.

## 5. Feasibility and Comparison with Current Grid Search

**Dimensionality Analysis:**
The number of degrees of freedom (DoF) managed by Differential Evolution depends exclusively on the asymmetric unit (Residue 0).
*   **Sites per residue**: 3 (C2, C3, C6).
*   **Rotatable bonds per selector**: Typically 2 for standard bundled carbamates (e.g., `tau_link`, `tau_ar`).
*   **Total Dihedrals ($D$)**: $3 \text{ sites} \times 2 \text{ dihedrals} = 6$.

A 6-dimensional continuous optimization problem is extremely tractable for Differential Evolution (DE).

**Cost Analysis: Current Greedy Grid Search vs. Differential Evolution**

1.  **Current Approach (Greedy Grid with Local Minimization):**
    *   **Evaluations:** Sweeps over sites and residues iteratively. Assuming 4 residues in the asymmetric repeat, 3 sites, `max_site_sweeps = 5`, and `max_candidates = 64` (from `basic.yaml`), a single optimization pass requires $5 \times 4 \times 3 \times 64 = 3,840$ evaluations. With `multi_opt.n_starts = 10`, this reaches $\sim 38,400$ evaluations.
    *   **Cost per Evaluation:** Each evaluation runs a two-stage local minimization (`Soft` + `Full` with ~120 total max iterations) on the full polymer in OpenMM. This is computationally heavy (tens to hundreds of milliseconds per evaluation).
    *   **Pros:** Explores discrete known basins; multi-start escapes some traps.
    *   **Cons:** Misses continuous global minima between grid points; scales poorly if grid resolution or selector flexibility increases; evaluates the whole polymer.

2.  **Proposed Approach (Differential Evolution):**
    *   **Evaluations:** DE requires a population size of typically $15 D$. For $D=6$, population = 90. Allowing for ~150 generations, the total evaluation count is roughly $90 \times 150 = 13,500$ evaluations.
    *   **Cost per Evaluation:** Each evaluation is a **single-point energy calculation** (`getState(getEnergy=True)`). No forces, no local minimization, and `Context.setPositions()` is incredibly fast. Constructing the coordinates is done analytically via NumPy array operations for the $N$ residues avoiding RDKit.
    *   **Pros:** Single-point energies take $\sim 1$ ms or less on modern hardware. 13,500 single-point evaluations will take merely $10{-}20$ seconds, outperforming the total runtime of the grid search drastically. It continuously samples the space, finding precisely interlocked conformations that grid discretization might miss.
    *   **Cons:** Gradient-free optimization requires careful bounding. High penalization for steric clashes ($10^9$ kJ/mol) must be robustly handled so the population correctly maps feasible space.

**Conclusion on Feasibility:**
Differential evolution is entirely feasible. Given the low dimensionality ($D=6$) for the repeating unit and the orders-of-magnitude faster single-point energy calculation compared to full gradient minimization, DE will likely be significantly faster and mathematically more rigorous in finding the true symmetry-coupled global minimum.

**Is Multi-Start Overkill for Differential Evolution?**

Unlike local minimizers (e.g., L-BFGS-B, conjugate gradient), Differential Evolution is a true *global* optimizer. By maintaining a diverse population of candidate solutions (the 90 vectors traversing the energy landscape), it natively avoids getting trapped in local funnels.

However, complex polymeric systems with bulky sidechains (like standard carbamate selectors) often have rugged energy surfaces featuring "golf-hole" minima—very deep but narrow energetic basins dictated by precise interlocking hydrogen bond networks. 

1.  **Single Run Convergence:** For a $D=6$ problem with a reasonably continuous landscape, a single DE run with an adequate population size and convergence tolerance strongly converges to the true global minimum.
2.  **The Case for Multi-Start DE:**
    *   **Ruggedness:** CSP selectors are notoriously rugged. DE relies on stochastic mutation and crossover. In highly rugged landscapes, the entire population might prematurely collapse into a broad but sub-optimal meta-basin, missing the narrow deep funnel entirely.
    *   **Cost-Benefit:** Because an entire DE run of 13,500 evaluations only takes $10-20$ seconds, running $N=5$ or $N=10$ independent starts using different integer random seeds is computationally trivial (1-2 minutes total on a single CPU core, or mere seconds if embarrassingly parallelized over cores).
    *   **Scientific Rigor:** Finding the *same* minimum energy state across multiple independent starts with different randomized initial populations is the gold standard for proving that the algorithm has actually discovered the true global minimum basin, rather than just settling in the first deep pocket it stumbled upon.

**Recommendation:**
Multi-start is **not overkill**. It is highly recommended. The extreme speed of single-point energy evaluations means we can afford to run 5-10 independent DE starts seeded with different random states. If all starts converge to the exact same $\mathbf{\Phi}_{opt}$ and energy, we gain immense statistical confidence in the derived conformer.

**Parallel vs. Sequential Execution of DE Starts:**

Currently, `multi_opt.py` uses `concurrent.futures.ProcessPoolExecutor` to map independent starts across available CPU cores. Should we keep this parallelism for Differential Evolution?

Let's look at the numbers. While a single DE start (13,500 single-point OpenMM `getEnergy=True` evaluations) is extremely fast ($\sim 10-20$ seconds), running $N=10$ starts sequentially would take $\sim 2-3$ minutes. 

*   **Sequential Pros:** Simplifies code logic; absolutely guarantees no OpenMM Context or multiprocessing context sharing issues; ideal for systems with limited physical cores where thread-thrashing might occur.
*   **Parallel Pros:** Reduces the $2-3$ minute wall-clock time back down to $10-20$ seconds (assuming $\ge 10$ cores). 

However, there is a fundamental bottleneck: **OpenMM GPU Initialization Overhead**.
Spawning a completely new `Process` and initializing a new OpenMM GPU `Context` from scratch takes roughly $\sim 3-5$ seconds per process. 
If a single DE run only actually computes for $10-20$ seconds, the fixed startup cost of spawning 10 parallel Python processes and 10 GPU contexts nearly negates the parallelization benefit, and may actually crash the GPU driver if it runs out of memory trying to allocate 10 simultaneous OpenMM contexts.

Therefore, for Differential Evolution where the individual start runtime is measured in seconds rather than minutes:
1.  **Run Sequentially by Default**: A single Python process can quickly cycle through 10 iterations of `differential_evolution` sequentially using the *exact same* pre-warmed OpenMM `Context`, simply passing a different integer `seed` to SciPy each time. This avoids $10\times$ GPU context creation overhead and prevents VRAM exhaustion. It will finish securely in $2-3$ minutes.
2.  **Optional Parallelism**: If the user has a massive generic cluster, they could use the existing CPU-based multiprocessing, but for standard workstation usage with a single GPU, sequential execution of multi-start DE is by far the safest, most robust, and practically identical in perceived wall-clock speed due to the elimination of startup overhead.

**Aggregating Multi-Start Results (Deduplication):**

Currently, `multi_opt.py` returns exactly `top_k` ranked results. If Differential Evolution strongly converges, it is highly likely that multiple independent starts will discover the exact same structure. Passing $K$ nearly identical structures down the pipeline to expensive stages like docking or AMBER descriptor export is a massive waste of computational resources.

To address this, the multi-start wrapper for DE should implement a **deduplicated aggregation** step before returning the final candidates.

**Proposal: Energy & RMSD Thresholding**
1.  After collecting the $N$ optimized conformers, sort them by final energy.
2.  Initialize an empty list of `unique_results`.
3.  For each sorted conformer, compare it against all currently accepted `unique_results`.
    *   If the energy difference $\Delta E$ is below a tight tolerance (e.g., $0.1$ kJ/mol) **AND** the heavy-atom RMSD (or the max absolute difference in $\mathbf{\Phi}$ space) is below a strict threshold (e.g., $5^\circ$), consider it a duplicate.
    *   If it is a duplicate, increment a `discovery_count` on the existing `unique_result` (useful metadata for assessing funnel depth/width) and discard the redundant conformer.
    *   If it is novel, append it to `unique_results`.
4.  Return only up to `top_k` of these *unique* results.

*   **Pros:** 
    *   **Resource Efficiency:** Drastically reduces downstream computing costs (docking, AMBER prep) by guaranteeing only structurally distinct minima are processed.
    *   **Statistical Insight:** The `discovery_count` reveals the landscape topological width. If a structure is found 9 out of 10 times, it implies an exceptionally broad, stable, global basin.
*   **Cons:** 
    *   Requires an RMSD or $\mathbf{\Phi}$-distance calculation, though for just $N$ final candidates, this takes negligible fractions of a millisecond.
    *   May return *fewer* than `top_k` results if the landscape is so structurally constrained that only 1 or 2 true minimum basins exist. (This is scientifically correct, but downstream scripts must handle receiving fewer outputs than requested).

## 6. Incorporating Non-Energetic Criteria

**Can we penalize structures with fewer H-bonds or stacked aromatic rings?**

Yes, and this maps perfectly onto the proposed architecture without sacrificing the extreme performance of Differential Evolution (DE). Since DE simply attempts to minimize a scalar objective function value (`score = evaluate_symmetric_energy(Phi)`), any arbitrary penalty or reward can be directly incorporated.

There are two primary paradigms for implementing this:

1.  **Native OpenMM Custom Forces (Recommended & Already Supported):**
    The `poly_csp` framework already supports translating structural criteria into energetic biases during the soft-optimization stage (e.g., `soft_selector_hbond_bias` and `anti_stacking_sigma_scale` in the `TwoStageMinimizationProtocol` / `PreparedRuntimeOptimizationBundle`).
    *   **How it works:** We evaluate the DE objective using the *soft-stage* OpenMM system rather than the bare full-stage physics. OpenMM inherently calculates the custom continuous pseudo-energies for H-bond rewards and aromatic-stacking repulsions directly on the GPU/C++ backend alongside the standard forcefield. 
    *   **Cost:** Essentially zero added overhead. The evaluation remains in the $\sim 1$ ms regime.

2.  **Post-Processing Penalties directly in Python:**
    If criteria cannot be easily expressed as continuous OpenMM forces (e.g., highly complex geometric constraints), we can compute them directly in the Python objective function using the $N \times 3$ NumPy coordinate array *after* it's been built but *before* calling OpenMM.
    *   **How it works:** We apply vectorized pairwise distance calculations in NumPy (e.g. using `scipy.spatial.distance.cdist`) to explicitly count valid donor-acceptor pair distances or detect $\pi$-$\pi$ coordinate overlaps. The final returned score becomes dynamically modified: `Score = OpenMM_Energy + Penalty(coords)`.
    *   **Cost:** Processing a few hundred atoms with vectorized NumPy adds perhaps $\sim 1-5$ ms per evaluation. While slower than pure OpenMM evaluations, it remains well within the feasible envelope. For the 13,500 evaluations needed, this adds at most 1 minute to the total run time.

**Recommendation:**
Adding non-energetic criteria is highly feasible. The best approach is to evaluate the DE population against the existing `soft` OpenMM system, which natively encapsulates structural constraints like anti-stacking and H-bond forcing as continuous energy biases, thus maintaining maximum evaluation throughput.

## 7. Periodic vs. Finite (Open) Systems

The `poly_csp` framework supports two distinct end modes for polymers: `periodic` (infinite) and `open` (finite). Symmetry ordering must handle both correctly while preserving the core mathematical design.

The proposed mathematical framework (Section 3) assumes a perfectly wrapped periodic system where atom $j$ on Residue 0 hydrogen-binds backwards across the periodic boundary to the last residue $N_{res}-1$. 

**How to Handle `end_mode="periodic"`**
*   **Implementation:** The algorithm described in Section 3 works out-of-the-box. We set the OpenMM `Context` to use periodic boundary conditions (`context.setPositions` using an explicitly set `DefaultPeriodicBoxVectors` where $Lz = dp \times \text{rise\_A}$).
*   **Symmetry Maintenance:** Because OpenMM natively wraps coordinates and handles cross-boundary interactions, the energy evaluated natively reflects an infinitely long chain. All $N$ residues simply clone their selectors from Residue 0.

**How to Handle `end_mode="open"` (Finite Polymers)**
For a finite polymer with open ends (e.g., $dp=24$), applying perfect periodic boundaries during optimization is physically incorrect because the real chain ends have vacuum boundaries and terminal hydroxyls/caps.

However, we **still want** to discover the mathematically perfect internal crystalline packing for the *center* of the chain, ignoring end-effects.

*   **Implementation Strategy (The "Periodic-Core" Proxy):** Even if the user requests an open $dp=24$ polymer, we should **not** evaluate the DE objective on the $dp=24$ open chain.
    1.  Instead, we temporarily extract the $N_{res}$ monomer repeating unit (e.g., $dp=4$ for amylose 4/3) and build a small proxy `periodic` OpenMM system in memory purely for the DE engine.
    2.  The DE engine optimizes this small $dp=4$ periodic system to find the optimal symmetric packing $\mathbf{\Phi}_{opt}$.
    3.  Once found, we simply apply $\mathbf{\Phi}_{opt}$ uniformly to *all* residues of the actual $dp=24$ open polymer.
*   **Pros:** 
    *   **Massive Speedup:** DE evaluates a tiny 4-residue system instead of 24 residues. 13,500 evaluations of a $dp=4$ system is extremely fast.
    *   **Pure Symmetry:** It guarantees the selectors are optimized for bulk symmetry, entirely uncorrupted by boundary effects (which can otherwise cause greedy optimizers to "unravel" the ends). The terminal selectors simply use the bulk optimal angles. 
*   **Cons:** 
    *   The very ends of the open chain might have slight steric clashes or suboptimal H-bonds because they lack a neighbor. *However*, this is mathematically identical to what happens in real life when a perfectly crystalline bulk phase terminates at a surface, and the subsequent downstream relaxation stage (`forcefield/options=runtime_relax`) will naturally allow those specific terminal selectors to relax away from the bulk symmetry during MD equilibration.

**Recommendation:**
The symmetry ordering algorithm should **always** operate on a small periodic proxy system ($dp=N_{res}$) regardless of whether the target `end_mode` is `periodic` or `open`. The discovered $\mathbf{\Phi}_{opt}$ is then mapped onto the full target topology. This guarantees maximum performance, enforces strictly pure bulk symmetry without end-effect corruption, and elegantly handles both end modes via a single mathematical path.

