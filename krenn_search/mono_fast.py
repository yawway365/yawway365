"""Fast class-level engine for mono-edge multigraphs (layers may overlap).

Variables: A[a, pair] complex, a in colors, pair in pairs(K_n).
w(chi) = sum over PMs M, colorings chi constant on each edge of M:
         prod_e A[chi(e), e].
Each PM touches exactly d^(n/2) colorings -> scatter-based evaluation.

Loss: sum_chi |w(chi) - [chi monochromatic]|^2, with analytic gradient.
"""
import numpy as np
from core import perfect_matchings, pairs


class MonoFast:
    def __init__(self, n, d):
        self.n, self.d = n, d
        self.pairs = pairs(n)
        self.pidx = {p: k for k, p in enumerate(self.pairs)}
        self.pms = perfect_matchings(n)
        self.m = n // 2
        P, m = len(self.pms), self.m
        self.pm_edges = np.array([[self.pidx[e] for e in M] for M in self.pms])
        # flat coloring index for each PM x color-combo
        # combo c = (c_0..c_{m-1}) colors per edge; coloring: vertex v gets
        # color of its edge; flat index = sum_v chi_v * d^(n-1-v)
        self.NT = d ** n
        vpow = d ** (n - 1 - np.arange(n))
        idx = np.zeros((P, d ** m), dtype=np.int64)
        combos = np.stack(np.meshgrid(*[np.arange(d)] * m, indexing="ij"),
                          axis=-1).reshape(-1, m)   # (d^m, m)
        for r, M in enumerate(self.pms):
            acc = np.zeros(d ** m, dtype=np.int64)
            for k, (i, j) in enumerate(M):
                acc += combos[:, k] * (vpow[i] + vpow[j])
            idx[r] = acc
        self.idx = idx
        self.combos = combos
        # target
        t = np.zeros(self.NT, dtype=complex)
        for a in range(d):
            t[int(a * (self.NT - 1) // (d - 1))] = 1.0  # a * (d^n-1)/(d-1) = a*sum d^k
        self.target = t
        # einsum labels
        self.letters = "abcde"[:m]

    def _pm_products(self, A):
        """(P, d^m) products for all PMs/combos."""
        P, m, d = len(self.pms), self.m, self.d
        vecs = [A[:, self.pm_edges[:, k]] for k in range(m)]  # each (d, P)
        sub = ",".join(f"{c}P" for c in self.letters) + "->P" + self.letters
        prods = np.einsum(sub, *vecs)  # (P, d,d,..,d)
        return prods.reshape(P, d ** m)

    def weight_flat(self, A):
        prods = self._pm_products(A)
        w = np.bincount(self.idx.ravel(), weights=prods.real.ravel(),
                        minlength=self.NT).astype(complex)
        w += 1j * np.bincount(self.idx.ravel(), weights=prods.imag.ravel(),
                              minlength=self.NT)
        return w

    def loss_and_grad(self, A):
        d, m, P = self.d, self.m, len(self.pms)
        prods = self._pm_products(A)
        w = np.bincount(self.idx.ravel(), weights=prods.real.ravel(),
                        minlength=self.NT).astype(complex)
        w += 1j * np.bincount(self.idx.ravel(), weights=prods.imag.ravel(),
                              minlength=self.NT)
        R = w - self.target
        loss = float(np.vdot(R, R).real)
        Rb = np.conj(R)[self.idx]           # (P, d^m)
        Rb = Rb.reshape((P,) + (d,) * m)
        G = np.zeros_like(A)
        vecs = [A[:, self.pm_edges[:, k]] for k in range(m)]
        for k in range(m):
            others = [vecs[j] for j in range(m) if j != k]
            lab = [self.letters[j] for j in range(m) if j != k]
            sub = "P" + self.letters + "," + \
                ",".join(f"{c}P" for c in lab) + "->" + self.letters[k] + "P"
            g = np.einsum(sub, Rb, *others)   # (d, P)
            np.add.at(G, (slice(None), self.pm_edges[:, k]), g)
        return loss, np.conj(G)

    def verify(self, A, tol=1e-9):
        w = self.weight_flat(A)
        err = np.abs(w - self.target)
        return float(err.max())
