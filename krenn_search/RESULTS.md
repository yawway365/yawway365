# Status report: hunting a d=4 counterexample to Krenn's conjecture

**Goal.** Find a *monochromatic* bi-colored weighted (multi)graph — Question 1
of Krenn–Gu–Soltész — with d = 4 colors: all 4 monochromatic inherited vertex
colorings get weight 1, every other coloring cancels out, on n ≥ 6 vertices
(n = 2, and (n,d) ∈ {(even,2),(4,3)} are the known/trivial cases). This would
disprove the Krenn(–Gu) conjecture (3 000 € prize, open since 2017).

**Bottom line so far.** No counterexample found. Every structured family we
examined is either provably incapable of exact cancellation or numerically
converges only to *asymptotic* pseudo-solutions (fidelity → 1 with the exact
system provably inconsistent). This matches the published evidence that the
conjecture is true; the impossibility results now cover d ≥ n (2026,
AlphaProof), n = 4 (2023), max-degree ≤ 3 (2024), vertex-connectivity ≤ 2,
mono-edge (n=6, d≥3) and (n=8, d≥4) (SAT, 2021), and positive weights (2017).
The smallest fully open exact cases are (n=6, d=3) and (n=6, d=4) with
bi-chromatic edges and complex weights.

## What we established (new, as far as we know)

### 1. The least-squares infimum for (n=6, d=3) is 0 — but it is not attained
Unconstrained gradient searches converge to sparse configurations of the form
"two monochromatic triangles + one cross perfect matching":

- triangle {0,1,2} with edge colors (01)->0, (02)->2, (12)->1,
- triangle {3,4,5} with edge colors (35)->0, (34)->2, (45)->1,
- cross matching (03)->1, (15)->2, (24)->0.

This support has exactly 4 perfect matchings: three monochromatic (one per
color) and one all-cross PM. Putting weight ε on the cross edges and ~1/√ε on
triangle edges keeps each mono class at 1 while the single bad class scales as
ε³ → 0. The exact system on this support requires x·y·z = 0 with x, y, z all
nonzero — inconsistent. Consequence: **naive least-squares/fidelity
optimization cannot decide the conjecture**; the monochromatic fidelity
supremum for (6,3) is 1 without being attained (vanishing postselection rate).

For (n=6, d=4) every run plateaus at loss = 1.0 (three colors alive via the
pseudo-family, fourth dead) — consistent with an infimum of 1 for d = 4.

### 2. Mono-edge layers built from 1-factorizations are dead even asymptotically
For n=6, d=4 with layers = 4 disjoint PMs of a 1-factorization of K6: the union
K6 minus a PM has exactly 8 PMs — the 4 layers plus 4 "bad" PMs, and the bad
PMs *partition* the 12 edges. Hence the sum of their log-weight magnitudes is
fixed to 0 by the mono normalizations: they cannot all be driven to 0, and the
4 bad colorings are distinct so no phase cancellation is possible either.

### 3. Path-uniqueness lemma (why most layer shapes cannot cancel)
If a color layer is a Hamiltonian cycle (or any disjoint union of cycles *used
as a graph*), every proper vertex subset induces disjoint paths, and a disjoint
union of paths has at most one perfect matching. Hence the induced coloring
determines the mixed PM uniquely — every mixed class is single-PM and forces a
hard zero. The same holds for bipartite "permutation + one extra cell" layers.
**Exact cancellation requires layers containing a C4 (a full 2×2 minor in the
bipartite case).** Empirically: random Ham-cycle layer structures on K10 had
~540 PMs with 100 % of mixed classes single-PM; random C4+C6 2-factor layers
still had ~90 % single-PM classes.

### 4. Bipartite factorization (clean reformulation)
For bipartite graphs (rows X, cols Y, m = n/2) with mono-edges given by layer
matrices B_1..B_d of pairwise disjoint support, the weight of the class
(α, β) *factorizes*:

    w(α, β) = ∏_a perm(B_a[α⁻¹(a), β⁻¹(a)])   (empty perm = 1).

So the monochromatic conditions become: perm(B_a) = 1 for all a, and every
nontrivial compatible partition must contain **one vanishing subpermanent**.
Cancellation is purely within a layer. Corollaries:
- permutation layers (Latin-square-like structures) reduce to transversal /
  compatible-partition combinatorics: for m=4, both order-4 Latin square types
  fail (Z2×Z2 has transversals; Z4 has coset splits) — matching the known
  matching-index results;
- at m = 5 (n = 10) a counting/tiling argument shows at most ONE layer can
  contain a 2×2 block: two block layers (7 cells each) always collide with the
  forced completion matchings — so (n=10, d=4) bipartite mono-edge has
  essentially a single scalar cancellation degree of freedom vs hundreds of
  active classes;
- the first roomy case is m = 6 (n = 12, d = 4): four block layers of 8 cells
  + 4 extras fit in the 36-cell grid. Randomized scans over these supports
  (622 active classes vs 36 complex weights in typical samples) have so far
  plateaued at cost ≈ 1.0.

## Negative numerical results (ongoing)
- Full complex-weight searches (analytic Wirtinger gradients + L-BFGS,
  basin-hopping): (6,3) → pseudo-solutions only; (6,4) → plateau 1.0.
- Doubling ansatz for (8,4): two fixed K4 d=3 blocks + 256 free complex cross
  weights (mono-3 from cross matchings): plateaus around loss ≈ 4.2.
- Bipartite (12,4) block-layer scans: plateau ≈ 1.0 so far.
- k-monochromatic searches (Question 2, with herald vertices; also
  prize-relevant per the Jan-2021 update): (n=6,k=4,d=4) and (n=8,k=4,d=4)
  running.

## Reproduce
    python3 test_known.py            # sanity: known d=2 and d=3 solutions
    python3 search.py 6 4 50         # full search (n=6, d=4)
    python3 campaign.py 6 3 1 0.2 t  # pseudo-solution family shows up quickly
    python3 bip_scan.py 6 4 20 1     # bipartite block-layer scan (n=12, d=4)

## Iteration 2 — Question-1-focused searches (in progress)

The k-monochromatic d=4 solution (SOLUTION_d4_kmono.md) does not satisfy
Question 1 (no heralds allowed), so the hunt continues on four fronts:

1. **Symmetry-constrained search** (`sym_search.py`): impose invariance under
   the joint rotation (v -> v+1 mod n, c -> c+1 mod d) at (n,d) = (8,4).
   All four mono classes become symmetry-equivalent, eliminating the
   "dead color" attractor that traps unconstrained runs at loss 1.0.
   58 orbit parameters instead of 448.
2. **Structure annealing** (`anneal_mono.py`): simulated annealing over
   mono-edge colored graphs at (10,4), minimizing missing-mono/single-PM-class
   counts, with inner least-squares solves on cancellation-ready structures.
3. **Full-space campaign** at (8,3) (`campaign.py`) — the smallest case not
   covered by any published impossibility result at d=3.
4. k-monochromatic (8,6,4) — would beat the Erhard graph in both k and d
   (side quest; not a Question-1 counterexample).

Structural notes from this iteration: for bipartite block+rest layers, all
(m-1,1)-type classes are structurally dead; the observed cost-1.0 plateau at
m=6 must come from richer class shapes (diagnosis pending).
