"""Symmetry-constrained search for a monochromatic graph (Question 1).

Impose invariance under the joint cyclic action
    vertices: v -> v+1 (mod n),   colors: c -> c+1 (mod d)
on the weight tensor W[(i,j)][a,b] (with the (i,j) pair reoriented so i<j;
when the rotation swaps the endpoint order, the color pair transposes).

All d mono classes are then symmetry-equivalent: no dead-color attractor.

Usage: python3 sym_search.py n d hours seed tag
"""
import sys
import time
import numpy as np
from scipy.optimize import minimize
from core import Engine


def build_orbits(n, d, pairs, pair_index):
    """Orbits of (pair_idx, a, b) under (i,j,a,b)->(i+1,j+1,a+1,b+1)."""
    seen = {}
    orbits = []
    for k, (i, j) in enumerate(pairs):
        for a in range(d):
            for b in range(d):
                if (k, a, b) in seen:
                    continue
                orb = []
                ci, cj, ca, cb = i, j, a, b
                while True:
                    ii, jj = (ci % n), (cj % n)
                    aa, bb = ca % d, cb % d
                    if ii < jj:
                        key = (pair_index[(ii, jj)], aa, bb)
                    else:
                        key = (pair_index[(jj, ii)], bb, aa)
                    if key in seen:
                        break
                    seen[key] = len(orbits)
                    orb.append(key)
                    ci, cj, ca, cb = ci + 1, cj + 1, ca + 1, cb + 1
                    if (ci % n, cj % n, ca % d, cb % d) == (i, j, a, b):
                        break
                if orb:
                    orbits.append(orb)
    return orbits


def main(n, d, hours, seed, tag):
    eng = Engine(n, d)
    orbits = build_orbits(n, d, eng.pairs, eng.pair_index)
    P = len(orbits)
    print(f"[{tag}] orbit parameters: {P} (full: {len(eng.pairs) * d * d})",
          flush=True)
    shape = (len(eng.pairs), d, d)
    rng = np.random.default_rng(seed)

    def expand(z):
        W = np.zeros(shape, dtype=complex)
        for oi, orb in enumerate(orbits):
            for (k, a, b) in orb:
                W[k, a, b] = z[oi]
        return W

    def collapse(G):
        g = np.zeros(P, dtype=complex)
        for oi, orb in enumerate(orbits):
            for (k, a, b) in orb:
                g[oi] += G[k, a, b]
        return g

    def f(x):
        z = x[:P] + 1j * x[P:]
        W = expand(z)
        L, G = eng.loss_and_grad(W)
        g = collapse(G)
        return L, np.concatenate([2 * g.real, 2 * g.imag])

    t_end = time.time() + hours * 3600
    best = np.inf
    r = 0
    while time.time() < t_end:
        scale = [0.25, 0.5, 0.9, 1.4][r % 4]
        x0 = rng.normal(scale=scale, size=2 * P)
        res = minimize(f, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 4000, "maxfun": 8000,
                                "ftol": 1e-18, "gtol": 1e-14})
        if res.fun < best:
            best = float(res.fun)
            np.save(f"{tag}_bestW.npy", expand(res.x[:P] + 1j * res.x[P:]))
        print(f"[{tag}] r={r} loss={res.fun:.6e} best={best:.6e}", flush=True)
        if best < 1e-16:
            print(f"*** [{tag}] CANDIDATE ***", flush=True)
            break
        r += 1


if __name__ == "__main__":
    n, d = int(sys.argv[1]), int(sys.argv[2])
    hours, seed, tag = float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    main(n, d, hours, seed, tag)
