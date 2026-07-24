"""Hafnian pencil relaxation (inspired by determinantal/Pfaffian
representations of Fermat hypersurfaces).

For mono-edge multigraph layers A_1..A_d (symmetric weighted adjacency on n
vertices), any exact monochromatic graph satisfies the polynomial identity

    Haf(t_1 A_1 + ... + t_d A_d) = t_1^{n/2} + ... + t_d^{n/2}.

The coefficient of t^kappa is the sum of all coloring-class weights whose
color-count vector is kappa — so this is a necessary (relaxed) condition.

Usage: python3 pencil.py n d restarts seed
"""
import sys
import itertools
import numpy as np
from scipy.optimize import least_squares
from core import perfect_matchings, pairs


class HafPencil:
    def __init__(self, n, d):
        self.n, self.d = n, d
        self.pairs = pairs(n)
        self.pidx = {p: k for k, p in enumerate(self.pairs)}
        self.pms = perfect_matchings(n)
        self.m = n // 2
        # monomials of degree m in d vars
        self.monos = [k for k in itertools.product(range(self.m + 1), repeat=d)
                      if sum(k) == self.m]
        self.midx = {k: i for i, k in enumerate(self.monos)}
        # partial monomials of degree <= m (DP states)
        self.states = [k for k in itertools.product(range(self.m + 1), repeat=d)
                       if sum(k) <= self.m]
        self.sidx = {k: i for i, k in enumerate(self.states)}
        # transition: state s + e_a -> state index (or -1)
        self.trans = np.full((len(self.states), d), -1, dtype=int)
        for s in self.states:
            for a in range(d):
                s2 = tuple(s[b] + (1 if b == a else 0) for b in range(d))
                if sum(s2) <= self.m:
                    self.trans[self.sidx[s], a] = self.sidx[s2]
        # PM edge index array (num_pms, m)
        self.pm_edges = np.array([[self.pidx[e] for e in M] for M in self.pms])
        # target
        self.target = np.zeros(len(self.monos), dtype=complex)
        for a in range(d):
            pure = tuple(self.m if b == a else 0 for b in range(d))
            self.target[self.midx[pure]] = 1.0
        # map final states (degree m) to monomial index
        self.final_map = np.full(len(self.states), -1, dtype=int)
        for k in self.monos:
            self.final_map[self.sidx[k]] = self.midx[k]

    def coeffs(self, A):
        """A: (d, num_pairs) complex layer weights -> coefficient vector."""
        P, m, d = len(self.pms), self.m, self.d
        S = len(self.states)
        dp = np.zeros((P, S), dtype=complex)
        dp[:, self.sidx[(0,) * d]] = 1.0
        for step in range(m):
            new = np.zeros_like(dp)
            eidx = self.pm_edges[:, step]           # (P,)
            for a in range(d):
                w = A[a, eidx]                       # (P,)
                # for every state with a valid transition under color a
                valid = self.trans[:, a] >= 0
                src = np.where(valid)[0]
                dst = self.trans[src, a]
                new[:, dst] += dp[:, src] * w[:, None]
            dp = new
        out = np.zeros(len(self.monos), dtype=complex)
        for si in range(S):
            mi = self.final_map[si]
            if mi >= 0:
                out[mi] += dp[:, si].sum()
        return out

    def residuals(self, A):
        return self.coeffs(A) - self.target

    def solve(self, restarts=10, seed=0, scale=0.6):
        rng = np.random.default_rng(seed)
        nv = self.d * len(self.pairs)

        def f(x):
            A = (x[:nv] + 1j * x[nv:]).reshape(self.d, len(self.pairs))
            r = self.residuals(A)
            return np.concatenate([r.real, r.imag])

        best = (np.inf, None)
        for r in range(restarts):
            x0 = rng.normal(scale=scale, size=2 * nv)
            sol = least_squares(f, x0, method="trf", max_nfev=1500)
            if sol.cost < best[0]:
                best = (sol.cost, (sol.x[:nv] + 1j * sol.x[nv:]).reshape(
                    self.d, len(self.pairs)))
            print(f"  restart {r}: cost={sol.cost:.6e} best={best[0]:.6e}",
                  flush=True)
            if best[0] < 1e-22:
                break
        return best


if __name__ == "__main__":
    n, d = int(sys.argv[1]), int(sys.argv[2])
    restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    hp = HafPencil(n, d)
    print(f"n={n} d={d}: {len(hp.monos)} coefficient conditions, "
          f"{d * len(hp.pairs)} complex unknowns, {len(hp.pms)} PMs")
    cost, A = hp.solve(restarts, seed)
    print("BEST COST:", cost)
    if cost < 1e-20 and A is not None:
        np.save(f"pencil_n{n}_d{d}.npy", A)
        print("saved pencil solution")
