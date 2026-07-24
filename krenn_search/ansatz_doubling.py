"""Doubling ansatz for (n=8, d=4):

V1={0..3}, V2={4..7} each carry the standard K4 d=3 monochromatic solution
on colors {0,1,2} (fixed).  All 16 cross pairs (u,v), u in V1, v in V2 carry
free 4x4 complex weight matrices (colors {0,1,2,3}).

Mono 0/1/2 arise from intra PM pairs; mono 3 must arise purely from cross
perfect matchings colored (3,3); all mixed classes must cancel.

We optimize only the cross entries with L-BFGS.

Usage: python3 ansatz_doubling.py restarts maxiter seed [tag]
"""
import sys
import numpy as np
from scipy.optimize import minimize
from core import Engine

N, D = 8, 4
K4_PMS = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]


def build_fixed_and_mask(eng, relax_intra=False):
    W0 = np.zeros((len(eng.pairs), D, D), dtype=complex)
    free = np.zeros((len(eng.pairs), D, D), dtype=bool)
    # intra K4 solutions
    for c, M in enumerate(K4_PMS):
        for (i, j) in M:
            W0[eng.pair_index[(i, j)]][c, c] = 1.0
            W0[eng.pair_index[(i + 4, j + 4)]][c, c] = 1.0
            if relax_intra:
                free[eng.pair_index[(i, j)]][c, c] = True
                free[eng.pair_index[(i + 4, j + 4)]][c, c] = True
    # cross pairs fully free
    for u in range(4):
        for v in range(4, 8):
            free[eng.pair_index[(u, v)]][:, :] = True
    return W0, free


def run(restarts, maxiter, seed, tag="doubling", relax_intra=False):
    eng = Engine(N, D)
    W0, free = build_fixed_and_mask(eng, relax_intra)
    nf = int(free.sum())
    rng = np.random.default_rng(seed)
    print(f"[{tag}] free complex params: {nf}", flush=True)

    def to_W(x):
        W = W0.copy()
        W[free] = x[:nf] + 1j * x[nf:]
        return W

    def f(x):
        W = to_W(x)
        L, G = eng.loss_and_grad(W)
        g = np.concatenate([2 * G[free].real, 2 * G[free].imag])
        return L, g

    best = (np.inf, None)
    for r in range(restarts):
        scale = [0.2, 0.5, 1.0][r % 3]
        x0 = rng.normal(scale=scale, size=2 * nf)
        # bias: seed the (3,3) cross entries of one matching near a solution of
        # prod = 1 and mixed entries near prod = -1
        res = minimize(f, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                                "ftol": 1e-18, "gtol": 1e-14})
        W = to_W(res.x)
        fid = eng.fidelity(W)
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
            np.save(f"{tag}_bestW.npy", W)
        print(f"[{tag}] r={r} loss={res.fun:.6e} fid={fid:.6f} best={best[0]:.6e}",
              flush=True)
        if best[0] < 1e-16:
            print(f"*** [{tag}] CANDIDATE ***", flush=True)
            break
    return best, eng


if __name__ == "__main__":
    restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tag = sys.argv[4] if len(sys.argv) > 4 else "doubling"
    relax = len(sys.argv) > 5 and sys.argv[5] == "relax"
    best, eng = run(restarts, maxiter, seed, tag, relax)
    print("BEST:", best[0])
