"""Sanity checks: the engine must confirm the two known monochromatic graphs."""
import numpy as np
from core import Engine


def test_c6_d2():
    n, d = 6, 2
    eng = Engine(n, d)
    W = np.zeros((len(eng.pairs), d, d), dtype=complex)
    # alternately colored 6-cycle 0-1-2-3-4-5-0, monochromatic edges
    cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    for k, e in enumerate(cycle):
        i, j = min(e), max(e)
        c = k % 2
        W[eng.pair_index[(i, j)]][c, c] = 1.0
    ok, err, _ = eng.verify(W)
    assert ok, err
    print("C6 d=2 OK")


def test_k4_d3():
    n, d = 4, 3
    eng = Engine(n, d)
    W = np.zeros((len(eng.pairs), d, d), dtype=complex)
    # 3 perfect matchings of K4 = color classes
    pms = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
    for c, M in enumerate(pms):
        for (i, j) in M:
            W[eng.pair_index[(i, j)]][c, c] = 1.0
    ok, err, _ = eng.verify(W)
    assert ok, err
    print("K4 d=3 OK")


def test_gradient():
    """Finite-difference check of the analytic gradient."""
    n, d = 4, 2
    eng = Engine(n, d)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(len(eng.pairs), d, d)) + 1j * rng.normal(size=(len(eng.pairs), d, d))
    L0, G = eng.loss_and_grad(W)
    eps = 1e-7
    for idx in [(0, 0, 0), (2, 1, 0), (5, 1, 1)]:
        for direction, part in [(1.0, "re"), (1j, "im")]:
            Wp = W.copy(); Wp[idx] += eps * direction
            L1, _ = eng.loss_and_grad(Wp)
            num = (L1 - L0) / eps
            ana = 2 * (G[idx].real if part == "re" else G[idx].imag)
            assert abs(num - ana) < 1e-4 * max(1, abs(ana)), (idx, part, num, ana)
    print("gradient OK")


if __name__ == "__main__":
    test_c6_d2()
    test_k4_d3()
    test_gradient()
