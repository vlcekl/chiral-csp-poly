# H-bond Network Connectivity

The exact connectivity of this hydrogen bond network is well-documented. Based on the foundational NMR, IR, and computational studies by Yoshio Okamoto and Eiji Yashima—the pioneers of these specific chiral stationary phases—the carbamate network follows a very strict, highly ordered topological blueprint.

Because the geometry of the polysaccharide backbone dictates the spacing, the hydrogen bonds strictly partition into two distinct structural roles: the **Adjacent Zipper** (C2 and C3) and the **Helical Pitch Bridge** (C6).

Here is the exact donor/acceptor connectivity you need for your algorithmic constraints:

### 1. The "Adjacent Zipper" (C2 and C3 Selectors)

The carbamates at positions C2 and C3 are physically restricted because they are attached directly to the rigid pyranose ring. They interact almost exclusively with the immediately adjacent glucose unit to lock the glycosidic bonds ($\phi$ and $\psi$) in place.

* **Connectivity:** The interactions cross over between the C2 and C3 positions of neighboring residues (Residue $i$ and Residue $i-1$).
* **Bond A:** The $N-H$ of the **C2** carbamate on Residue $i$ donates a hydrogen bond to the $C=O$ of the **C3** carbamate on the adjacent Residue $i-1$.
* **Bond B:** Conversely, the $N-H$ of the **C3** carbamate on Residue $i$ donates to the $C=O$ of the **C2** carbamate on Residue $i-1$.
* **Result:** This creates a continuous, interlocking "cross-stitch" or zipper that runs the entire length of the polymer backbone, making the helix incredibly stiff.

### 2. The "Helical Pitch Bridge" (C6 Selector)

The C6 carbamate is attached to the exocyclic methylene group ($-CH_2-$), giving it a longer, more flexible "arm." It does *not* bond with the adjacent residue. Instead, it reaches vertically up or down the central axis of the polymer to bond with the selector sitting exactly one helical turn away.

* **Connectivity:** The interaction occurs strictly between **C6 and C6**. The $N-H$ of a C6 carbamate donates to the $C=O$ of the C6 carbamate on the residue situated one full helical pitch away.
* **For Amylose (AD, AY, etc.):** Because amylose forms a left-handed 4/3 helix (4 units per 3 turns), Residue 1 and Residue 5 align vertically. The C6 hydrogen bond forms between **Residue $i$ and Residue $i+4$**.
* **For Cellulose (IB, IC, etc.):** Because cellulose forms a tighter left-handed 3/2 helix (3 units per 2 turns), Residue 1 and Residue 4 align vertically. The C6 hydrogen bond forms between **Residue $i$ and Residue $i+3$**.

### How to use this in your new algorithm:

When you build your objective function or distance constraints for the new script, you can now explicitly target these exact index pairs.

Instead of a generic "find any H-bond" reward, you can mathematically penalize the optimizer if:

1. The C2($N$) of Unit 2 is not within 3.0 Å of the C3($O$) of Unit 1.
2. The C6($N$) of Unit 5 is not within 3.0 Å of the C6($O$) of Unit 1 (for an amylose pentamer seed).

This deterministic mapping is the "secret sauce" that will allow your algorithm to instantly snap the selectors into the correct, globally symmetric orientation without relying on stochastic MD to find it.