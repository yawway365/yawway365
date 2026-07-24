"""Numerical search for a monochromatic bi-colored graph (Krenn Question 1).

Minimizes  L(W) = sum_chi |w(chi) - delta_mono(chi)|^2  over the full complex
weight space (num_pairs x d x d complex), with L-BFGS-B and analytic
Wirtinger gradients, from many random restarts.  A counterexample to the
conjecture corresponds to L -> 0 for n >= 6, d >= 3.

Usage: python3 search.py n d restarts [maxiter] [seed]
"""
import sys
import json
import numpy as np
from scipy.optimize import minimize
from core import Engine


def pack(W):
    return np.concatenate([W.real.ravel(), W.imag.ravel()])


def unpack(x, shape):
    half = x.size // 2
    return x[:half].reshape(shape) + 1j * x[half:].reshape(shape)


def run(n, d, restarts, maxiter=3000, seed=0, scale=0.7, out=None):
    eng = Engine(n, d)
    shape = (len(eng.pairs), d, d)
    rng = np.random.default_rng(seed)

    def f(x):
        W = unpack(x, shape)
        L, G = eng.loss_and_grad(W)
        g = np.concatenate([2 * G.real.ravel(), 2 * G.imag.ravel()])
        return L, g

    best = {"loss": np.inf}
    for r in range(restarts):
        x0 = rng.normal(scale=scale, size=2 * np.prod(shape))
        res = minimize(f, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": maxiter * 2,
                                "ftol": 1e-18, "gtol": 1e-14})
        W = unpack(res.x, shape)
        fid = eng.fidelity(W)
        if res.fun < best["loss"]:
            best = {"loss": float(res.fun), "fidelity": fid, "restart": r,
                    "W": W.copy()}
        print(f"[n={n} d={d}] restart {r:3d}: loss={res.fun:.6e}  fid={fid:.6f}  "
              f"best_loss={best['loss']:.6e} best_fid={best['fidelity']:.6f}",
              flush=True)
        if best["loss"] < 1e-16:
            print("*** LOSS ~ 0 : CANDIDATE COUNTEREXAMPLE FOUND ***", flush=True)
            break
    if out:
        np.save(out + "_W.npy", best["W"])
        meta = {k: v for k, v in best.items() if k != "W"}
        meta.update({"n": n, "d": d, "restarts": restarts, "seed": seed})
        with open(out + "_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
    return best, eng


if __name__ == "__main__":
    n = int(sys.argv[1]); d = int(sys.argv[2]); restarts = int(sys.argv[3])
    maxiter = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out = sys.argv[6] if len(sys.argv) > 6 else f"best_n{n}_d{d}_s{seed}"
    best, eng = run(n, d, restarts, maxiter, seed, out=out)
    print(json.dumps({k: v for k, v in best.items() if k != "W"}, indent=2))
    eng.verify(best["W"], tol=1e-8)
