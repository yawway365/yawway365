"""Optimized bipartite mono-edge search (see bipartite.py for the math).

Key speedups:
- subpermanents computed only over valid (Xmask,Ymask) pairs (C(2m,m) total);
- residuals restricted to structurally-active classes (all factors have a
  support transversal) -- inactive classes are identically zero;
- class products vectorized via precomputed gather indices.
"""
import itertools
import numpy as np
from scipy.optimize import least_squares


def popcount(x):
    return bin(x).count("1")


class FastBipartite:
    def __init__(self, m, d, support):
        self.m, self.d = m, d
        self.support = dict(support)
        self.cells = sorted(support)
        self.cidx = {c: k for k, c in enumerate(self.cells)}
        self.layer_cells = [[c for c in self.cells if support[c] == a]
                            for a in range(d)]
        # ordered valid pairs (X,Y), X ascending so recursion is ready
        self.pairs = [(X, Y) for X in range(1 << m) for Y in range(1 << m)
                      if popcount(X) == popcount(Y)]
        self.pidx = {p: k for k, p in enumerate(self.pairs)}
        # recursion plan per pair: list of (y, child_pair_index)
        self.full = (1 << m) - 1
        self.plan = []
        for (X, Y) in self.pairs:
            if X == 0:
                self.plan.append(None)
                continue
            x0 = (X & -X).bit_length() - 1
            Xr = X & ~(1 << x0)
            steps = []
            for y in range(m):
                if Y >> y & 1:
                    steps.append((x0, y, self.pidx[(Xr, Y & ~(1 << y))]))
            self.plan.append(steps)
        self._build_classes()

    def _build_classes(self):
        m, d = self.m, self.d
        parts = []
        for coloring in itertools.product(range(d), repeat=m):
            masks = [0] * d
            for i, a in enumerate(coloring):
                masks[a] |= 1 << i
            parts.append(tuple(masks))
        bysizes = {}
        for p in parts:
            key = tuple(popcount(x) for x in p)
            bysizes.setdefault(key, []).append(p)
        # structural support subperms (0/1)
        ind = [self._subperms(self._layer_matrix(a, structural=True))
               for a in range(d)]
        self.active = []
        for key, rps in bysizes.items():
            if m in key:
                continue  # monochromatic
            for rp in rps:
                for cp in bysizes[key]:
                    idxs = [self.pidx[(rp[a], cp[a])] for a in range(d)]
                    if all(abs(ind[a][idxs[a]]) > 1e-9 for a in range(d)):
                        self.active.append(idxs)
        self.active = np.array(self.active, dtype=int) if self.active else \
            np.zeros((0, d), dtype=int)

    def _layer_matrix(self, a, w=None, structural=False):
        B = np.zeros((self.m, self.m), dtype=complex)
        for c in self.layer_cells[a]:
            B[c] = 1.0 if structural else w[self.cidx[c]]
        return B

    def _subperms(self, B):
        SP = np.zeros(len(self.pairs), dtype=complex)
        for k, p in enumerate(self.pairs):
            steps = self.plan[k]
            if steps is None:
                SP[k] = 1.0
                continue
            s = 0j
            for (x0, y, child) in steps:
                v = B[x0, y]
                if v != 0:
                    s += v * SP[child]
            SP[k] = s
        return SP

    def residuals(self, w):
        SPs = [self._subperms(self._layer_matrix(a, w)) for a in range(self.d)]
        fidx = self.pidx[(self.full, self.full)]
        res = [SPs[a][fidx] - 1.0 for a in range(self.d)]
        if len(self.active):
            prod = np.ones(len(self.active), dtype=complex)
            for a in range(self.d):
                prod *= SPs[a][self.active[:, a]]
            res = np.concatenate([np.array(res), prod])
        else:
            res = np.array(res)
        return res

    def n_active(self):
        return len(self.active)

    def solve(self, restarts=20, seed=0, scale=1.0, max_nfev=2500):
        rng = np.random.default_rng(seed)
        k = len(self.cells)

        def f(x):
            r = self.residuals(x[:k] + 1j * x[k:])
            return np.concatenate([r.real, r.imag])

        best = (np.inf, None)
        for _ in range(restarts):
            x0 = rng.normal(scale=scale, size=2 * k)
            sol = least_squares(f, x0, method="trf", max_nfev=max_nfev)
            if sol.cost < best[0]:
                best = (sol.cost, sol.x[:k] + 1j * sol.x[k:])
            if best[0] < 1e-24:
                break
        return best


def random_block_support(m, d, rng, n_extra=0):
    """Each layer: one 2x2 block + completion to a support with a transversal,
    all layers disjoint.  Returns dict or None."""
    cells_free = {(i, j) for i in range(m) for j in range(m)}
    support = {}
    for a in range(d):
        placed = False
        for _ in range(60):
            rows = rng.choice(m, 2, replace=False)
            cols = rng.choice(m, 2, replace=False)
            block = [(int(rows[0]), int(cols[0])), (int(rows[0]), int(cols[1])),
                     (int(rows[1]), int(cols[0])), (int(rows[1]), int(cols[1]))]
            if all(c in cells_free for c in block):
                # complete with a partial permutation on remaining rows/cols
                rrows = [r for r in range(m) if r not in rows]
                rcols = [c for c in range(m) if c not in cols]
                perm = list(rng.permutation(len(rcols)))
                rest = [(rrows[i], rcols[perm[i]]) for i in range(len(rrows))]
                if all(c in cells_free for c in rest):
                    for c in block + rest:
                        support[c] = a
                        cells_free.discard(c)
                    placed = True
                    break
        if not placed:
            return None
    # optionally sprinkle extra free cells into random layers
    for _ in range(n_extra):
        if not cells_free:
            break
        c = list(cells_free)[rng.integers(len(cells_free))]
        support[c] = int(rng.integers(d))
        cells_free.discard(c)
    return support
