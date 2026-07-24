"""Search for k-monochromatic graphs (Question 2 of Krenn-Gu-Soltész):
first k vertices share a color a (all d choices, unit weight), remaining
n-k vertices are red (color 0); every other coloring cancels.

The Erhard graph (n=8, k=6, d=3) is the only known example with k>4, d>=3.
Target here: d=4.

Usage: python3 kmono_search.py n k d hours seed tag
"""
import sys
import time
import json
import numpy as np
from scipy.optimize import minimize
from core import Engine


def make_target(n, k, d):
    t = np.zeros((d,) * n, dtype=complex)
    for a in range(d):
        idx = tuple([a] * k + [0] * (n - k))
        t[idx] = 1.0
    return t


def main(n, k, d, hours, seed, tag):
    eng = Engine(n, d)
    eng.target = make_target(n, k, d)
    shape = (len(eng.pairs), d, d)
    rng = np.random.default_rng(seed)

    def f(x):
        half = x.size // 2
        W = x[:half].reshape(shape) + 1j * x[half:].reshape(shape)
        L, G = eng.loss_and_grad(W)
        return L, np.concatenate([2 * G.real.ravel(), 2 * G.imag.ravel()])

    t_end = time.time() + hours * 3600
    best_loss, best_x = np.inf, None
    r = 0
    while time.time() < t_end:
        scale = [0.3, 0.7, 1.2][r % 3]
        x0 = rng.normal(scale=scale, size=2 * int(np.prod(shape)))
        res = minimize(f, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000, "maxfun": 6000,
                                "ftol": 1e-18, "gtol": 1e-14})
        if res.fun < best_loss:
            best_loss, best_x = float(res.fun), res.x.copy()
            half = best_x.size // 2
            np.save(f"{tag}_bestW.npy",
                    best_x[:half].reshape(shape) + 1j * best_x[half:].reshape(shape))
        print(f"[{tag}] r={r} loss={res.fun:.6e} best={best_loss:.6e}", flush=True)
        if best_loss < 1e-16:
            print(f"*** [{tag}] SOLUTION FOUND ***", flush=True)
            break
        r += 1
    with open(f"{tag}_summary.json", "w") as fh:
        json.dump({"n": n, "k": k, "d": d, "best_loss": best_loss,
                   "restarts": r}, fh, indent=2)


if __name__ == "__main__":
    n, k, d = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    hours = float(sys.argv[4]); seed = int(sys.argv[5]); tag = sys.argv[6]
    main(n, k, d, hours, seed, tag)
