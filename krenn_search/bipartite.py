"""Bipartite mono-edge search via subpermanent factorization.

Vertices = rows X (m) + cols Y (m), n = 2m.  Layer a = complex matrix B_a,
supports pairwise disjoint inside the m x m grid.  For a vertex coloring
(alpha on rows, beta on cols) the coloring-class weight factorizes:

    w(alpha,beta) = prod_a perm(B_a[X_a, Y_a]),
    X_a = alpha^{-1}(a), Y_a = beta^{-1}(a)   (empty perm = 1;
    |X_a| != |Y_a| -> 0 structurally).

Monochromatic conditions: perm(B_a) = 1 for all a; every mixed compatible
class must vanish, i.e. some factor's subpermanent = 0.
"""
import itertools
import numpy as np


def popcount(x):
    return bin(x).count("1")


class BipartiteSystem:
    def __init__(self, m, d, support):
        """support: dict {(x,y): color a}."""
        self.m, self.d = m, d
        self.support = dict(support)
        self.cells = sorted(support)
        self.cidx = {c: k for k, c in enumerate(self.cells)}
        # per layer: cell list
        self.layer_cells = [[c for c in self.cells if support[c] == a]
                            for a in range(d)]
        # subset pairs with equal popcount
        self.pairs_eq = [(X, Y) for X in range(1 << m) for Y in range(1 << m)
                         if popcount(X) == popcount(Y)]
        # enumerate compatible classes: (alpha, beta) as tuples of masks
        # per color; represent class by tuple of (Xmask_a, Ymask_a).
        self.classes = self._enumerate_classes()

    def _enumerate_classes(self):
        m, d = self.m, self.d
        # all ordered set partitions of [m] into d labeled (possibly empty) blocks
        parts = []  # list of tuple-of-masks
        for coloring in itertools.product(range(d), repeat=m):
            masks = [0] * d
            for i, a in enumerate(coloring):
                masks[a] |= 1 << i
            parts.append(tuple(masks))
        bysizes = {}
        for p in parts:
            key = tuple(popcount(x) for x in p)
            bysizes.setdefault(key, []).append(p)
        classes = []
        for key, rows in bysizes.items():
            for rp in rows:
                for cp in bysizes[key]:
                    # skip the d monochromatic classes
                    if any(popcount(rp[a]) == self.m for a in range(self.d)):
                        continue
                    classes.append((rp, cp))
        return classes

    # ---------- permanents ----------
    def all_subperms(self, B):
        """B: m x m complex.  Returns dict {(Xmask,Ymask): perm} for equal
        popcount pairs (empty = 1)."""
        m = self.m
        SP = {(0, 0): 1.0 + 0j}
        # iterate masks by popcount
        for X in range(1 << m):
            for Y in range(1 << m):
                if X == 0:
                    continue
                if popcount(X) != popcount(Y):
                    continue
                x0 = (X & -X).bit_length() - 1  # lowest row in X
                s = 0j
                Xr = X & ~(1 << x0)
                for y in range(m):
                    if Y >> y & 1:
                        v = B[x0, y]
                        if v != 0:
                            s += v * SP[(Xr, Y & ~(1 << y))]
                SP[(X, Y)] = s
        return SP

    def layer_matrices(self, w):
        Bs = []
        for a in range(self.d):
            B = np.zeros((self.m, self.m), dtype=complex)
            for c in self.layer_cells[a]:
                B[c] = w[self.cidx[c]]
            Bs.append(B)
        return Bs

    def residuals(self, w):
        Bs = self.layer_matrices(w)
        SPs = [self.all_subperms(B) for B in Bs]
        full = (1 << self.m) - 1
        res = [SPs[a][(full, full)] - 1.0 for a in range(self.d)]
        for (rp, cp) in self.classes:
            v = 1.0 + 0j
            for a in range(self.d):
                v *= SPs[a][(rp[a], cp[a])]
                if v == 0:
                    break
            res.append(v)
        return np.array(res, dtype=complex)

    def active_classes(self):
        """Classes whose factors are all structurally nonzero (support of each
        submatrix has a transversal).  These are the ones weights must kill."""
        Bs = self.layer_matrices(np.ones(len(self.cells), dtype=complex))
        SPs = [self.all_subperms(np.abs(B)) for B in Bs]
        act = []
        for (rp, cp) in self.classes:
            if all(abs(SPs[a][(rp[a], cp[a])]) > 1e-12 for a in range(self.d)):
                act.append((rp, cp))
        return act

    def solve(self, restarts=20, seed=0, scale=1.0):
        from scipy.optimize import least_squares
        rng = np.random.default_rng(seed)
        k = len(self.cells)

        def f(x):
            r = self.residuals(x[:k] + 1j * x[k:])
            return np.concatenate([r.real, r.imag])

        best = (np.inf, None)
        for _ in range(restarts):
            x0 = rng.normal(scale=scale, size=2 * k)
            sol = least_squares(f, x0, method="trf", max_nfev=3000)
            if sol.cost < best[0]:
                best = (sol.cost, sol.x[:k] + 1j * sol.x[k:])
            if best[0] < 1e-24:
                break
        return best
