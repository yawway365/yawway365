"""Core engine for Krenn's "monochromatic graph" question (Question 1 of
Krenn, Gu, Soltész, "Questions on the Structure of Perfect Matchings
inspired by Quantum Physics").

Formalism
---------
A bi-colored weighted (multi)graph on n vertices with d colors is fully
described (after merging parallel edges with identical color pairs) by a
d x d complex matrix W[(i,j)] for every unordered pair i<j:

    W[(i,j)][a, b] = weight of the edge between i and j colored a at i, b at j.

For a vertex coloring chi in {0..d-1}^n the weight of the coloring is

    w(chi) = sum over perfect matchings M of K_n of
             prod over (i,j) in M of W[(i,j)][chi_i, chi_j].

G is *monochromatic* (a counterexample to the Krenn conjecture when n>=6 and
d>=3) iff w(chi) = 1 for all d constant colorings and w(chi) = 0 otherwise.
"""

import itertools
import numpy as np


def perfect_matchings(n):
    """All perfect matchings of K_n as tuples of (i,j) pairs, i<j."""
    verts = tuple(range(n))

    def rec(vs):
        if not vs:
            yield ()
            return
        i = vs[0]
        rest = vs[1:]
        for k, j in enumerate(rest):
            for tail in rec(rest[:k] + rest[k + 1:]):
                yield ((i, j),) + tail

    return list(rec(verts))


def pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


class Engine:
    """Vectorized computation of the full weight tensor w(chi) (shape d^n)
    and Wirtinger gradients of the least-squares loss."""

    def __init__(self, n, d):
        self.n, self.d = n, d
        self.pairs = pairs(n)
        self.pair_index = {p: k for k, p in enumerate(self.pairs)}
        self.pms = perfect_matchings(n)
        # target tensor: 1 on the d constant colorings, 0 elsewhere
        t = np.zeros((d,) * n, dtype=complex)
        for a in range(d):
            t[(a,) * n] = 1.0
        self.target = t
        # einsum subscripts per matching
        letters = "abcdefghijklmnopqrstuvwxyz"
        self.vert_sub = letters[:n]
        self.pm_subs = []
        for M in self.pms:
            subs = ",".join(self.vert_sub[i] + self.vert_sub[j] for (i, j) in M)
            self.pm_subs.append(subs + "->" + self.vert_sub)

    def weight_tensor(self, W):
        """W: array (num_pairs, d, d) complex -> tensor of w(chi), shape (d,)*n."""
        d, n = self.d, self.n
        out = np.zeros((d,) * n, dtype=complex)
        for M, sub in zip(self.pms, self.pm_subs):
            mats = [W[self.pair_index[p]] for p in M]
            out += np.einsum(sub, *mats)
        return out

    def loss_and_grad(self, W):
        """Least-squares loss sum_chi |w(chi)-target|^2 and gradient dL/d(conj W)
        (so real gradient wrt Re/Im is 2*Re(g), 2*Im(g))."""
        d, n = self.d, self.n
        w = self.weight_tensor(W)
        R = w - self.target
        loss = float(np.vdot(R, R).real)
        G = np.zeros_like(W)
        Rc = np.conj(R)
        vs = self.vert_sub
        for M in self.pms:
            for (i, j) in M:
                others = [p for p in M if p != (i, j)]
                in_subs = [vs] + [vs[a] + vs[b] for (a, b) in others]
                mats = [Rc] + [W[self.pair_index[p]] for p in others]
                sub = ",".join(in_subs) + "->" + vs[i] + vs[j]
                # contract residual with the other matrices, summing all
                # vertex axes except i and j
                G[self.pair_index[(i, j)]] += np.einsum(sub, *mats)
        # dL/dWbar = conj( dL/dW ); G computed above is conj(residual)*prod(others)
        # which equals dL/dW; return its conjugate for gradient wrt conj vars.
        return loss, np.conj(G)

    def fidelity(self, W):
        """Monochromatic fidelity F = |sum_mono w|^2 / (d * sum |w|^2)."""
        w = self.weight_tensor(W)
        N = float(np.vdot(w, w).real)
        if N == 0:
            return 0.0
        s = sum(w[(a,) * self.n] for a in range(self.d))
        return float(abs(s) ** 2 / (self.d * N))

    # ---------- verification ----------
    def verify(self, W, tol=1e-9, verbose=True):
        """Check the monochromatic conditions; returns (ok, max_err, report)."""
        w = self.weight_tensor(W)
        errs = np.abs(w - self.target)
        max_err = float(errs.max())
        ok = max_err < tol
        report = {
            "max_error": max_err,
            "mono_weights": {a: complex(w[(a,) * self.n]) for a in range(self.d)},
            "num_nonzero_offmono": int(
                np.count_nonzero(
                    (np.abs(w) > tol) & (np.abs(self.target) == 0)
                )
            ),
        }
        if verbose:
            print(f"n={self.n} d={self.d}  max|w(chi)-target| = {max_err:.3e}  ok={ok}")
            for a, v in report["mono_weights"].items():
                print(f"  w(mono color {a}) = {v:.6f}")
            print(f"  off-mono colorings with |w|>{tol}: {report['num_nonzero_offmono']}")
        return ok, max_err, report
