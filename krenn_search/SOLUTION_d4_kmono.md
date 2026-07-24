# An exact d=4 k-monochromatic graph (n=6, k=4) — verified

This is a solution with **d = 4 colors** to Question 2 (k-monochromatic
graphs) of Krenn–Gu–Soltész, *Questions on the Structure of Perfect Matchings
inspired by Quantum Physics* — the heralded variant of the monochromatic-graph
question. In quantum terms: a linear-optics + pair-source experiment that
produces a **4-dimensional 4-photon GHZ state heralded by 2 ancilla photons**.
The previously highest dimension appearing in a known k-monochromatic graph
with k ≥ 4 was d = 3 (the K4 graph, and the Erhard graph for k = 6).

It is **not** a counterexample to the Krenn conjecture (Question 1), which
requires no herald vertices; see RESULTS.md for the extensive negative
evidence we gathered on that.

## The graph

n = 6 vertices; vertices 0–3 carry the state, vertices 4, 5 are heralds.
Colors {0 = red, 1, 2, 3}. All 17 edges are monochromatic (same color at both
endpoints) and all weights are rational:

| edge | color | weight |
|------|-------|--------|
| (0,1) | 0 | −1/2 |
| (0,2) | 0 | −1/2 |
| (1,3) | 0 | 1/2 |
| (2,3) | 0 | 1/2 |
| (0,4) | 0 | 1 |
| (3,4) | 0 | −1 |
| (4,5) | 0 | 2 |
| (0,5) | 0 | 1 |
| (1,5) | 0 | 1 |
| (2,5) | 0 | 1 |
| (3,5) | 0 | 1 |
| (0,2) | 1 | 1/2 |
| (1,3) | 1 | 1 |
| (0,1) | 2 | 1/2 |
| (2,3) | 2 | 1 |
| (0,3) | 3 | 1/2 |
| (1,2) | 3 | 1 |

Note the red layer omits edge (1,2) — that absence is load-bearing.

## Why it works (hand proof sketch)

Because every edge is monochromatic, a coloring class is determined by which
vertex sets take which colors, and its weight factorizes into per-layer
matching sums. Layers 1–3 are the three perfect matchings of K4 on vertices
0–3 (colored 1, 2, 3), so the only nonzero classes are: the four
k-monochromatic classes, and classes "one colored edge e + red on the rest".
Writing F(S) for the red-layer matching sum on vertex set S, the conditions
are:

- k-mono, color a ∈ {1,2,3}:  (1/2 · 1) · F({4,5}) = (1/2)·2 = 1  ✓
- k-mono, color 0 (all red):  F({0,…,5}) = (−1)·? … total = 1  ✓ (direct
  enumeration: contributions via herald pairings (4,5), (0,4), (3,4) give
  −1 + 1 + 1 = 1)
- for each colored edge e, F(complement of e) = 0:
  - e=(0,2): F({1,3,4,5}) = (1/2)(2) + (−1)(1) = 0
  - e=(1,3): F({0,2,4,5}) = (−1/2)(2) + (1)(1) = 0
  - e=(0,1): F({2,3,4,5}) = (1/2)(2) + (−1)(1) = 0
  - e=(2,3): F({0,1,4,5}) = (−1/2)(2) + (1)(1) = 0
  - e=(0,3): F({1,2,4,5}) = 0 (no red matching exists: (1,2) red edge absent)
  - e=(1,2): F({0,3,4,5}) = 0·2 + (1)(1) + (1)(−1) = 0

## Verification

Two independent implementations confirm all 4^6 = 4096 coloring classes:

    python3 - <<'PY'
    import sympy as sp
    from verify_exact import check_kmono
    R = sp.Rational
    weights = {
        (0,1,0,0): R(-1,2), (0,2,0,0): R(-1,2),
        (1,3,0,0): R(1,2),  (2,3,0,0): R(1,2),
        (0,4,0,0): 1, (3,4,0,0): -1, (4,5,0,0): 2,
        (0,5,0,0): 1, (1,5,0,0): 1, (2,5,0,0): 1, (3,5,0,0): 1,
        (0,2,1,1): R(1,2), (1,3,1,1): 1,
        (0,1,2,2): R(1,2), (2,3,2,2): 1,
        (0,3,3,3): R(1,2), (1,2,3,3): 1,
    }
    assert check_kmono(6, 4, 4, weights)
    PY

Output: `EXACT VERIFICATION PASSED: n=6 k=4 d=4; 4 unit classes, all others
cancel.`

## How it was found

Gradient search (L-BFGS with analytic Wirtinger gradients) over the full
complex weight space with the k-monochromatic target converged to loss
~1e-16 with bounded weights; iterative support pruning reduced it to 18
mono-edge entries; the resulting red-layer cancellation equations were then
solved by hand over the rationals (choosing gauge t=2, a=1, b=−1), shrinking
the support to 17.
