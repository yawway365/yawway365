"""Long-running restart campaign with varied init scales and basin hopping.

Usage: python3 campaign.py n d seed hours [tag]
"""
import sys
import time
import json
import numpy as np
from scipy.optimize import minimize
from core import Engine
from search import pack, unpack


def main(n, d, seed, hours, tag):
    eng = Engine(n, d)
    shape = (len(eng.pairs), d, d)
    rng = np.random.default_rng(seed)

    def f(x):
        W = unpack(x, shape)
        L, G = eng.loss_and_grad(W)
        return L, np.concatenate([2 * G.real.ravel(), 2 * G.imag.ravel()])

    def polish(x0, maxiter=2500):
        return minimize(f, x0, jac=True, method="L-BFGS-B",
                        options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                                 "ftol": 1e-18, "gtol": 1e-14})

    t_end = time.time() + hours * 3600
    best_loss, best_x = np.inf, None
    hist = []
    r = 0
    while time.time() < t_end:
        mode = r % 4
        if mode == 3 and best_x is not None:
            # basin hop: perturb current best
            sigma = 10 ** rng.uniform(-2, 0)
            x0 = best_x + rng.normal(scale=sigma, size=best_x.size)
        else:
            scale = [0.3, 0.7, 1.5][mode]
            x0 = rng.normal(scale=scale, size=2 * int(np.prod(shape)))
        res = polish(x0)
        W = unpack(res.x, shape)
        fid = eng.fidelity(W)
        if res.fun < best_loss - 1e-12:
            best_loss, best_x = float(res.fun), res.x.copy()
            np.save(f"{tag}_best_W.npy", W)
        hist.append(round(float(res.fun), 6))
        print(f"[{tag}] r={r} mode={mode} loss={res.fun:.6e} fid={fid:.6f} "
              f"best={best_loss:.6e}", flush=True)
        if best_loss < 1e-16:
            print(f"*** [{tag}] CANDIDATE FOUND ***", flush=True)
            break
        r += 1
    with open(f"{tag}_summary.json", "w") as fh:
        json.dump({"n": n, "d": d, "seed": seed, "restarts": r,
                   "best_loss": best_loss,
                   "loss_histogram": sorted(set(hist))[:50]}, fh, indent=2)
    print(f"[{tag}] done: {r} restarts, best loss {best_loss:.6e}", flush=True)


if __name__ == "__main__":
    n, d, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    hours = float(sys.argv[4])
    tag = sys.argv[5] if len(sys.argv) > 5 else f"camp_n{n}_d{d}_s{seed}"
    main(n, d, seed, hours, tag)
