"""Simulated annealing over mono-edge colored graph structures.

State: dict {(i,j): color}.  Energy (lexicographic, scalarized):
  1000 * (#missing mono classes) + 10 * (#single-PM mixed classes)
  + 0.05 * (#mixed classes)
Structures reaching zero singles get an inner least-squares solve; any
inner cost < 1e-20 would be a counterexample candidate (then verified
exactly).

Usage: python3 anneal_mono.py n d hours seed tag
"""
import sys
import time
import json
import numpy as np
from mono_edge import MonoEdgeSystem, graph_pms


def layer_has_pm(n, edges):
    """Backtracking: does graph (n, edges) have a perfect matching?"""
    adj = {v: set() for v in range(n)}
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)

    def rec(uncov):
        if not uncov:
            return True
        v = min(uncov)
        for u in adj[v]:
            if u in uncov:
                if rec(uncov - {v, u}):
                    return True
        return False

    return rec(frozenset(range(n)))


def energy(n, d, colored):
    sysm = MonoEdgeSystem(n, d, colored)
    s = sysm.summary()
    missing = d - s["mono_classes_present"]
    e = 1000.0 * missing + 10.0 * s["mixed_single_pm"] + 0.05 * s["mixed_classes"]
    return e, s, sysm


def initial_state(n, d, rng):
    """d random even 2-factors with a C4 (reuse factor_search generator)."""
    from factor_search import random_structure
    ctype = (4, n - 4) if n > 4 else (4,)
    st = None
    while st is None:
        st = random_structure(n, d, ctype, rng)
    return st


def main(n, d, hours, seed, tag):
    rng = np.random.default_rng(seed)
    allpairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    state = initial_state(n, d, rng)
    E, s, _ = energy(n, d, state)
    best = (E, dict(state))
    t_end = time.time() + hours * 3600
    T0, T1 = 8.0, 0.4
    it = 0
    solved_log = []
    while time.time() < t_end:
        frac = min(1.0, it / 20000.0)
        T = T0 * (T1 / T0) ** frac
        cand = dict(state)
        mv = rng.random()
        if mv < 0.45 and cand:
            e = list(cand)[rng.integers(len(cand))]
            c_new = int(rng.integers(d + 1))
            if c_new == d:
                del cand[e]
            else:
                cand[e] = c_new
        elif mv < 0.9:
            e = allpairs[rng.integers(len(allpairs))]
            cand[e] = int(rng.integers(d))
        else:
            if cand:
                e = list(cand)[rng.integers(len(cand))]
                del cand[e]
        # every layer must keep a PM
        ok = all(layer_has_pm(n, [e for e, c in cand.items() if c == a])
                 for a in range(d))
        if not ok:
            it += 1
            continue
        E2, s2, sysm2 = energy(n, d, cand)
        if E2 <= E or rng.random() < np.exp((E - E2) / T):
            state, E, s = cand, E2, s2
            if E < best[0]:
                best = (E, dict(state))
                print(f"[{tag}] it={it} E={E:.2f} {s}", flush=True)
                with open(f"{tag}_best.json", "w") as fh:
                    json.dump({"E": E, "state": {str(k): v for k, v in state.items()}}, fh)
            if s["mixed_single_pm"] == 0 and s["mono_classes_present"] == d:
                cost, w = sysm2.solve(restarts=6, seed=int(rng.integers(1e9)))
                solved_log.append(cost)
                print(f"[{tag}] it={it} ZERO-SINGLES structure, inner cost={cost:.4e}",
                      flush=True)
                if cost < 1e-20:
                    print(f"*** [{tag}] COUNTEREXAMPLE CANDIDATE ***", flush=True)
                    with open(f"{tag}_HIT.json", "w") as fh:
                        json.dump({"state": {str(k): v for k, v in state.items()},
                                   "w_re": list(w.real), "w_im": list(w.imag)}, fh)
                    return
        it += 1
    print(f"[{tag}] done. best E={best[0]:.2f}, inner costs seen: "
          f"{sorted(solved_log)[:10]}", flush=True)


if __name__ == "__main__":
    n, d = int(sys.argv[1]), int(sys.argv[2])
    hours, seed, tag = float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    main(n, d, hours, seed, tag)
