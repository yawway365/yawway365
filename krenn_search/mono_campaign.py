"""Class-level mono-edge campaign using MonoFast.

Usage: python3 mono_campaign.py n d hours seed tag [pencil_seed.npy]
If a pencil seed file (shape (d, num_pairs)) is given, half the restarts start
from it plus noise — the pencil manifold satisfies the necessary Fermat
coefficient conditions.
"""
import sys
import time
import json
import numpy as np
from scipy.optimize import minimize
from mono_fast import MonoFast


def main(n, d, hours, seed, tag, seedfile=None):
    mf = MonoFast(n, d)
    shape = (d, len(mf.pairs))
    rng = np.random.default_rng(seed)
    seedA = np.load(seedfile) if seedfile else None

    def f(x):
        half = x.size // 2
        A = x[:half].reshape(shape) + 1j * x[half:].reshape(shape)
        L, G = mf.loss_and_grad(A)
        return L, np.concatenate([2 * G.real.ravel(), 2 * G.imag.ravel()])

    t_end = time.time() + hours * 3600
    best_loss, r = np.inf, 0
    while time.time() < t_end:
        if seedA is not None and r % 2 == 0:
            noise = rng.normal(scale=0.1, size=shape) + \
                1j * rng.normal(scale=0.1, size=shape)
            A0 = seedA + noise
            x0 = np.concatenate([A0.real.ravel(), A0.imag.ravel()])
        else:
            x0 = rng.normal(scale=0.5, size=2 * int(np.prod(shape)))
        res = minimize(f, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 1500, "maxfun": 3000,
                                "ftol": 1e-18, "gtol": 1e-14})
        if res.fun < best_loss:
            best_loss = float(res.fun)
            half = res.x.size // 2
            np.save(f"{tag}_bestA.npy",
                    res.x[:half].reshape(shape) + 1j * res.x[half:].reshape(shape))
        print(f"[{tag}] r={r} loss={res.fun:.6e} best={best_loss:.6e}", flush=True)
        if best_loss < 1e-16:
            print(f"*** [{tag}] CANDIDATE ***", flush=True)
            break
        r += 1
    with open(f"{tag}_summary.json", "w") as fh:
        json.dump({"n": n, "d": d, "restarts": r, "best_loss": best_loss}, fh)


if __name__ == "__main__":
    n, d = int(sys.argv[1]), int(sys.argv[2])
    hours, seed, tag = float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    seedfile = sys.argv[6] if len(sys.argv) > 6 else None
    main(n, d, hours, seed, tag, seedfile)
