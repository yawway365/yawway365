"""Randomized structural search over mono-edge candidates built from
Hamiltonian-cycle decompositions of K_n.

K_{2m} decomposes into (m-1) Hamiltonian cycles + 1 perfect matching
(round-robin / Walecki).  For n=10 that is exactly 4 cycles -> d=4 layers.

Usage: python3 structure_search.py n d iters seed
"""
import sys
import itertools
import numpy as np
from mono_edge import MonoEdgeSystem


def walecki(n):
    """Standard decomposition of K_n (n even) into n/2-1 Ham cycles + 1 PM.
    Round-robin construction on vertices 0..n-2 circle + n-1 center."""
    m = n - 1
    cycles = []
    for r in range((n - 2) // 2):
        # rotation r: cycle through pairs of the round-robin schedule
        # standard zigzag hamiltonian cycle construction
        seq = [n - 1]
        cur = r
        seq.append(cur)
        for k in range(1, n - 1):
            if k % 2 == 1:
                cur = (r + (k + 1) // 2) % m
            else:
                cur = (r - k // 2) % m
            seq.append(cur)
        cyc = [(seq[i], seq[(i + 1) % n]) for i in range(n)]
        cycles.append([(min(a, b), max(a, b)) for (a, b) in cyc])
    # leftover PM
    used = set(e for c in cycles for e in c)
    allp = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pm = [e for e in allp if e not in used]
    return cycles, pm


def check_decomposition(n):
    cycles, pm = walecki(n)
    used = [e for c in cycles for e in c] + pm
    assert len(used) == len(set(used)) == n * (n - 1) // 2, "not a decomposition"
    for c in cycles:
        deg = {}
        for (a, b) in c:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        assert all(v == 2 for v in deg.values()), "layer not 2-regular"
    assert len(pm) == n // 2
    return cycles, pm


def random_structure(n, d, rng):
    """Colored edges: relabel vertices randomly, take d of the Ham cycles as
    color layers (drop the rest and the PM)."""
    cycles, pm = walecki(n)
    perm = rng.permutation(n)
    colored = {}
    ids = rng.permutation(len(cycles))[:d]
    for a, ci in enumerate(ids):
        for (i, j) in cycles[ci]:
            u, v = int(perm[i]), int(perm[j])
            colored[(min(u, v), max(u, v))] = a
    return colored


def main(n, d, iters, seed):
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for it in range(iters):
        colored = random_structure(n, d, rng)
        sysm = MonoEdgeSystem(n, d, colored)
        s = sysm.summary()
        if s["mono_classes_present"] < d:
            continue
        if sysm.obviously_infeasible():
            status = "infeasible-prefilter"
            print(f"it={it} {s} {status}", flush=True)
            continue
        cost, w = sysm.solve(restarts=8, seed=int(rng.integers(1e9)))
        print(f"it={it} {s} cost={cost:.6e}", flush=True)
        if cost < best[0]:
            best = (cost, colored, w)
        if cost < 1e-22:
            print("*** EXACT-LOOKING SOLUTION ***")
            print(colored)
            print(w)
            break
    print("best cost:", best[0])
    return best


if __name__ == "__main__":
    n, d, iters, seed = (int(x) for x in sys.argv[1:5])
    check_decomposition(n)
    main(n, d, iters, seed)
