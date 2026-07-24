"""Independent exact verifier for bi-colored weighted graphs.

Given weights as a dict {(i, j, a, b): sympy expression} (i<j, colors a at i
and b at j), checks the k-monochromatic (or monochromatic, k=n) conditions by
direct enumeration of all perfect matchings of K_n and all d^n colorings,
entirely in exact arithmetic. No numpy, no shared code with core.py.
"""
import itertools
import sympy as sp


def perfect_matchings(verts):
    verts = list(verts)
    if not verts:
        yield []
        return
    i = verts[0]
    for k in range(1, len(verts)):
        j = verts[k]
        rest = verts[1:k] + verts[k + 1:]
        for tail in perfect_matchings(rest):
            yield [(i, j)] + tail


def coloring_weights(n, d, weights):
    """Returns dict {coloring tuple: exact weight}."""
    out = {}
    pms = list(perfect_matchings(list(range(n))))
    for chi in itertools.product(range(d), repeat=n):
        total = sp.Integer(0)
        for M in pms:
            term = sp.Integer(1)
            for (i, j) in M:
                w = weights.get((i, j, chi[i], chi[j]), 0)
                if w == 0:
                    term = sp.Integer(0)
                    break
                term *= w
            total += term
        total = sp.simplify(sp.expand(total))
        if total != 0:
            out[chi] = total
    return out


def check_kmono(n, k, d, weights, verbose=True):
    """All d colorings (a,)*k + (0,)*(n-k) must have weight 1; others 0."""
    got = coloring_weights(n, d, weights)
    targets = {tuple([a] * k + [0] * (n - k)): sp.Integer(1) for a in range(d)}
    ok = True
    for chi, val in got.items():
        want = targets.get(chi, sp.Integer(0))
        if sp.simplify(val - want) != 0:
            ok = False
            if verbose:
                print("MISMATCH", chi, "got", val, "want", want)
    for chi, want in targets.items():
        if chi not in got:
            ok = False
            if verbose:
                print("MISSING", chi, "want", want)
    if verbose and ok:
        print(f"EXACT VERIFICATION PASSED: n={n} k={k} d={d}; "
              f"{len(targets)} unit classes, all others cancel.")
    return ok
