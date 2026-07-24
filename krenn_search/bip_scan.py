"""Scan random bipartite supports; solve the promising ones.

Usage: python3 bip_scan.py m d iters seed [solve_threshold]
Layer budget for (m=5,d=4): two block layers (7), one near-perm (6), one perm (5).
For (m=4,d=3): two block layers (6), one perm (4).
"""
import sys
import numpy as np
from bipartite_fast import FastBipartite


def gen_support(m, d, rng, outer_tries=60):
    for _ in range(outer_tries):
        supp = _gen_support_once(m, d, rng)
        if supp is not None:
            return supp
    return None


def _gen_support_once(m, d, rng):
    cells_free = {(i, j) for i in range(m) for j in range(m)}

    def find_matching(rows, cols):
        """Backtracking perfect matching rows->cols using free cells, random order."""
        rows = list(rows)
        if not rows:
            return []
        r = rows[0]
        options = [c for c in cols if (r, c) in cells_free]
        rng.shuffle(options)
        for c in options:
            rest = find_matching(rows[1:], [x for x in cols if x != c])
            if rest is not None:
                return [(r, c)] + rest
        return None

    def place_perm(a, extra=0):
        cells = find_matching(list(range(m)), list(range(m)))
        if cells is None:
            return False
        for c in cells:
            support[c] = a
            cells_free.discard(c)
        for _ in range(extra):
            free = [c for c in cells_free]
            if not free:
                break
            c = free[rng.integers(len(free))]
            support[c] = a
            cells_free.discard(c)
        return True

    def place_block(a):
        for _ in range(200):
            rows = sorted(int(x) for x in rng.choice(m, 2, replace=False))
            cols = sorted(int(x) for x in rng.choice(m, 2, replace=False))
            block = [(r, c) for r in rows for c in cols]
            if not all(c in cells_free for c in block):
                continue
            rrows = [r for r in range(m) if r not in rows]
            rcols = [c for c in range(m) if c not in cols]
            rest = find_matching(rrows, rcols)
            if rest is not None:
                for c in block + rest:
                    support[c] = a
                    cells_free.discard(c)
                return True
        return False

    support = {}
    if (m, d) == (5, 4):
        plan = [("block",), ("block",), ("perm", 1), ("perm", 0)]
    elif (m, d) == (4, 3):
        plan = [("block",), ("block",), ("perm", 0)]
    elif (m, d) == (6, 4):
        plan = [("block",), ("block",), ("block",), ("block",)]
    elif (m, d) == (6, 3):
        plan = [("block",), ("block",), ("block",)]
    elif (m, d) == (7, 4):
        plan = [("block",), ("block",), ("block",), ("block",)]
    else:
        raise ValueError("no plan")
    # place block layers first (most constrained), then permutation layers
    order = sorted(range(d), key=lambda a: 0 if plan[a][0] == "block" else 1)
    for a in order:
        p = plan[int(a)]
        ok = place_block(int(a)) if p[0] == "block" else \
            place_perm(int(a), p[1] if len(p) > 1 else 0)
        if not ok:
            return None
    # distribute leftover cells as extras into random layers (they add
    # further cancellation structure and never hurt feasibility)
    free = sorted(cells_free)
    for c in free:
        support[c] = int(rng.integers(d))
        cells_free.discard(c)
    return support


def main(m, d, iters, seed, thresh=60):
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    hist = {}
    for it in range(iters):
        supp = gen_support(m, d, rng)
        if supp is None:
            continue
        sysm = FastBipartite(m, d, supp)
        na = sysm.n_active()
        hist[na] = hist.get(na, 0) + 1
        if na > thresh:
            continue
        cost, w = sysm.solve(restarts=8, seed=int(rng.integers(1e9)))
        print(f"it={it} active={na} cost={cost:.6e}", flush=True)
        if cost < best[0]:
            best = (cost, supp, w)
            if cost < 1e-20:
                print("*** EXACT-LOOKING SOLUTION ***", flush=True)
                print(supp)
                print(repr(w))
                break
    print("active histogram:", dict(sorted(hist.items())[:20]))
    print("best:", best[0])
    if best[1]:
        print("best support:", best[1])
        print("best weights:", repr(best[2]))
    return best


if __name__ == "__main__":
    m, d, iters, seed = (int(x) for x in sys.argv[1:5])
    thresh = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    main(m, d, iters, seed, thresh)
