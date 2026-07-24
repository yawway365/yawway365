"""Structural search over mono-edge bi-colored graphs.

A mono-edge candidate is an assignment color: E -> {0..d-1} on a graph
H on n vertices (each edge one color, one complex weight w_e).
Induced coloring of a PM M: vertex v gets the color of its M-edge.

Exact monochromatic conditions:
  for each color a: sum of prod(w_e) over PMs with all edges color a
                    AND covering all vertices = 1
  for every other induced coloring class: sum of prod(w_e) = 0

The key point: the constraint system only involves PMs of H, so for sparse
H it is small.  We solve it numerically (least squares over C with random
restarts) and, on success, exactly (groebner) on the discovered support.
"""
import itertools
import numpy as np
from scipy.optimize import least_squares


def graph_pms(n, edges):
    """Perfect matchings of graph (n, edges); edges = list of (i,j), i<j."""
    adj = {v: [] for v in range(n)}
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)
    out = []

    def rec(uncov, acc):
        if not uncov:
            out.append(tuple(acc))
            return
        v = min(uncov)
        for u in adj[v]:
            if u in uncov and u != v:
                e = (min(v, u), max(v, u))
                uncov2 = uncov - {v, u}
                rec(uncov2, acc + [e])

    rec(frozenset(range(n)), [])
    return out


class MonoEdgeSystem:
    def __init__(self, n, d, colored_edges):
        """colored_edges: dict {(i,j): color}."""
        self.n, self.d = n, d
        self.edges = sorted(colored_edges)
        self.color = dict(colored_edges)
        self.eidx = {e: k for k, e in enumerate(self.edges)}
        self.pms = graph_pms(n, self.edges)
        # group PMs by induced coloring
        groups = {}
        for M in self.pms:
            chi = [None] * n
            for e in M:
                chi[e[0]] = chi[e[1]] = self.color[e]
            groups.setdefault(tuple(chi), []).append(M)
        self.groups = groups
        self.mono = [tuple([a] * n) for a in range(d)]

    def realizes_all_colors(self):
        return all(c in self.groups for c in self.mono)

    def residuals(self, w):
        """w complex vector over edges -> residual per coloring class."""
        res = []
        for chi, Ms in self.groups.items():
            s = sum(np.prod([w[self.eidx[e]] for e in M]) for M in Ms)
            t = 1.0 if chi in self.mono else 0.0
            res.append(s - t)
        # missing mono colorings count as residual 1
        for c in self.mono:
            if c not in self.groups:
                res.append(1.0 + 0j)
        return np.array(res, dtype=complex)

    def solve(self, restarts=30, seed=0, scale=1.0):
        rng = np.random.default_rng(seed)
        m = len(self.edges)

        def f(x):
            w = x[:m] + 1j * x[m:]
            r = self.residuals(w)
            return np.concatenate([r.real, r.imag])

        nres = 2 * (len(self.groups) + sum(1 for c in self.mono if c not in self.groups))
        method = "lm" if nres >= 2 * m else "trf"
        best = (np.inf, None)
        for _ in range(restarts):
            x0 = rng.normal(scale=scale, size=2 * m)
            sol = least_squares(f, x0, method=method, max_nfev=4000)
            if sol.cost < best[0]:
                best = (sol.cost, sol.x[:m] + 1j * sol.x[m:])
            if best[0] < 1e-24:
                break
        return best  # (cost = 0.5*sum residual^2, weights)

    def summary(self):
        mixed = [c for c in self.groups if c not in self.mono]
        cancellable = sum(1 for c in mixed if len(self.groups[c]) > 1)
        singles = len(mixed) - cancellable
        return {
            "edges": len(self.edges), "pms": len(self.pms),
            "mono_classes_present": sum(1 for c in self.mono if c in self.groups),
            "mixed_classes": len(mixed),
            "mixed_single_pm": singles,  # each forces a hard zero (product=0)
        }

    def obviously_infeasible(self):
        """A mixed coloring class realized by exactly ONE PM forces some edge
        weight to 0; propagate: any edge that must be zero and is needed for a
        mono class kills the structure.  Quick necessary check: if some mixed
        class has 1 PM whose edges all appear in mono-class PMs, infeasible
        (unless a mono class has an alternative PM avoiding them - checked
        conservatively)."""
        # conservative: return True only if a single-PM mixed class exists whose
        # every edge is essential (appears in every PM of some mono class)
        for chi, Ms in self.groups.items():
            if chi in self.mono or len(Ms) != 1:
                continue
            M = Ms[0]
            # solvable only if we can zero one edge of M without killing a
            # mono class entirely
            ok = False
            for e in M:
                kills = False
                for c in self.mono:
                    if c in self.groups:
                        if all(e in M2 for M2 in self.groups[c]):
                            kills = True
                            break
                if not kills:
                    ok = True
                    break
            if not ok:
                return True
        return False
