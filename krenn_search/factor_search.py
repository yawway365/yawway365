"""Structural search v2: layers are even 2-factors with short cycles
(cycle type e.g. C4+C6 on n=10), so that layer subgraphs induced on proper
vertex subsets can have >= 2 matchings -> mixed coloring classes with >= 2
PMs -> cancellation is possible at all.

Usage: python3 factor_search.py n d iters seed [cycletypes]
  cycletypes: comma list like "4-6" meaning each layer is C4+C6
"""
import sys
import itertools
import numpy as np
from mono_edge import MonoEdgeSystem


def find_cycle(avail_adj, verts, length, rng, tries=200):
    """Random cycle of given length on a subset of verts using avail edges."""
    verts = list(verts)
    for _ in range(tries):
        start = verts[rng.integers(len(verts))]
        path = [start]
        used = {start}
        ok = True
        while len(path) < length:
            nxt = [u for u in avail_adj[path[-1]] if u not in used and u in verts]
            if len(path) == length - 1:
                nxt = [u for u in nxt if start in avail_adj[u]]
            if not nxt:
                ok = False
                break
            u = nxt[rng.integers(len(nxt))]
            path.append(u)
            used.add(u)
        if ok:
            return [(min(a, b), max(a, b))
                    for a, b in zip(path, path[1:] + [path[0]])]
    return None


def find_2factor(n, avail, ctype, rng, tries=300):
    """Random 2-factor with cycle lengths ctype (sums to n) from avail edges."""
    adj = {v: set() for v in range(n)}
    for (i, j) in avail:
        adj[i].add(j)
        adj[j].add(i)
    for _ in range(tries):
        remaining = set(range(n))
        a2 = {v: set(adj[v]) for v in range(n)}
        factor = []
        ok = True
        for L in ctype:
            cyc = find_cycle(a2, remaining, L, rng)
            if cyc is None:
                ok = False
                break
            factor += cyc
            for (i, j) in cyc:
                pass
            cyc_verts = set(v for e in cyc for v in e)
            remaining -= cyc_verts
            for v in cyc_verts:
                for u in list(a2[v]):
                    a2[v].discard(u)
                    a2[u].discard(v)
        if ok:
            return factor
    return None


def random_structure(n, d, ctype, rng, tries=50):
    allp = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for _ in range(tries):
        avail = set(allp)
        colored = {}
        ok = True
        for a in range(d):
            f = find_2factor(n, avail, ctype, rng)
            if f is None:
                ok = False
                break
            for e in f:
                colored[e] = a
                avail.discard(e)
        if ok:
            return colored
    return None


def main(n, d, iters, seed, ctype):
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    stats = {}
    for it in range(iters):
        colored = random_structure(n, d, ctype, rng)
        if colored is None:
            print("generation failed", flush=True)
            continue
        sysm = MonoEdgeSystem(n, d, colored)
        s = sysm.summary()
        stats.setdefault(s["mixed_single_pm"], 0)
        stats[s["mixed_single_pm"]] += 1
        if s["mono_classes_present"] < d or s["mixed_single_pm"] > 0:
            if it % 20 == 0:
                print(f"it={it} {s} skip  singles_hist={dict(sorted(stats.items())[:6])}",
                      flush=True)
            continue
        cost, w = sysm.solve(restarts=6, seed=int(rng.integers(1e9)))
        print(f"it={it} {s} SOLVE cost={cost:.6e}", flush=True)
        if cost < best[0]:
            best = (cost, colored, w)
        if cost < 1e-22:
            print("*** EXACT-LOOKING SOLUTION ***")
            print(colored)
            print(w)
            break
    print("singles histogram:", dict(sorted(stats.items())))
    print("best cost:", best[0])
    if best[1] is not None:
        print("best structure:", best[1])
    return best


if __name__ == "__main__":
    n, d, iters, seed = (int(x) for x in sys.argv[1:5])
    ctype = tuple(int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "4-6").split("-"))
    assert sum(ctype) == n
    main(n, d, iters, seed, ctype)
