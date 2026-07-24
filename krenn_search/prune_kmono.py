"""Iteratively prune the (6,4,4) k-mono solution support, checkpointing.
Usage: python3 prune_kmono.py W_in.npy tag
"""
import sys
import numpy as np
from scipy.optimize import least_squares
from core import Engine
from kmono_search import make_target

eng = Engine(6, 4)
eng.target = make_target(6, 4, 4)
W = np.load(sys.argv[1])
tag = sys.argv[2]


def solve_on(mask, x_init=None, seed=0):
    def f(x):
        Wx = np.zeros_like(W)
        half = x.size // 2
        Wx[mask] = x[:half] + 1j * x[half:]
        r = (eng.weight_tensor(Wx) - eng.target).ravel()
        return np.concatenate([r.real, r.imag])
    if x_init is None:
        rng = np.random.default_rng(seed)
        x_init = rng.normal(size=2 * int(mask.sum()))
    return least_squares(f, x_init, method="lm", max_nfev=8000,
                         xtol=1e-15, ftol=1e-15)


mask = np.abs(W) > 1e-3
sol = solve_on(mask, np.concatenate([W[mask].real, W[mask].imag]))
cur_mask, cur_x = mask, sol.x

def save(mask, x):
    Wx = np.zeros_like(W)
    half = x.size // 2
    Wx[mask] = x[:half] + 1j * x[half:]
    np.save(f"{tag}_minW.npy", Wx)

save(cur_mask, cur_x)
while True:
    half = cur_x.size // 2
    Wx = np.zeros_like(W)
    Wx[cur_mask] = cur_x[:half] + 1j * cur_x[half:]
    vals = np.abs(Wx[cur_mask])
    order = np.argsort(vals)
    pruned = False
    for oi in order[:8]:
        idxs = np.argwhere(cur_mask)
        cand = cur_mask.copy()
        cand[tuple(idxs[oi])] = False
        keep = np.ones(len(idxs), dtype=bool)
        keep[oi] = False
        xw = np.concatenate([cur_x[:half][keep], cur_x[half:][keep]])
        sol = solve_on(cand, xw)
        ok = sol.cost < 1e-24
        if not ok:
            for sd in range(4):
                sol = solve_on(cand, seed=sd)
                if sol.cost < 1e-24:
                    ok = True
                    break
        if ok:
            cur_mask, cur_x, pruned = cand, sol.x, True
            save(cur_mask, cur_x)
            print("pruned to", int(cand.sum()), "cost", sol.cost, flush=True)
            break
    if not pruned:
        break
print("FINAL support:", int(cur_mask.sum()))
half = cur_x.size // 2
Wx = np.zeros_like(W)
Wx[cur_mask] = cur_x[:half] + 1j * cur_x[half:]
for k, p in enumerate(eng.pairs):
    ent = [(a, b, Wx[k, a, b]) for a in range(4) for b in range(4)
           if cur_mask[k, a, b]]
    for a, b, v in ent:
        print(f"edge {p} colors ({a},{b}): {v:.8f}")
