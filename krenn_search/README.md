# Search for a counterexample to Krenn's conjecture (inherited vertex coloring)

Target: a bi-colored weighted (multi)graph on n vertices with d >= 3 colors
(n >= 6) in which all d monochromatic inherited vertex colorings have weight 1
and every other coloring cancels out (Question 1 of Krenn-Gu-Soltész,
https://mariokrenn.wordpress.com/graph-theory-question/). A d=4 instance
suffices.

## Files
- `core.py`      - weight-tensor engine, Wirtinger gradients, verifier
- `test_known.py`- sanity checks against the known d=2 (C6) and d=3 (K4) graphs
- `search.py`    - L-BFGS least-squares search over the full complex weight space
- `campaign.py`  - long-running restart/basin-hopping campaigns
- `mono_edge.py` - structural (mono-edge) candidate systems: PM enumeration,
                   coloring classes, small least-squares solve
- `structure_search.py` - randomized search over Hamiltonian-cycle layer
                   decompositions of K_n (n=10 gives exactly 4 layers)
- `ansatz_doubling.py`  - (n=8,d=4) ansatz: two fixed K4 d=3 blocks + free
                   bipartite cross weights; mono color 3 from cross matchings

## Findings so far
- (n=6,d=3): least-squares infimum is 0 but NOT attained: optimizer converges
  to an asymptotic family (two mono triangles + cross matching; bad PM weight
  eps^3 -> 0 while triangle weights ~ 1/sqrt(eps)). The exact system on that
  support is inconsistent (needs x*y*z=0 with x,y,z != 0). Exact equality is
  the actual open question; naive least squares cannot certify it.
- (n=6,d=4): all runs plateau at loss 1.0 (fidelity 3/4): three colors via the
  pseudo-family, fourth color dead.
- mono-edge (n=6,d=4) with layers = 4 disjoint PMs of a 1-factorization:
  provably impossible even asymptotically (the 4 bad PMs of K6-minus-PM
  partition the 12 edges, so their log-weight sums add to 0 and cannot all be
  driven to -inf).
