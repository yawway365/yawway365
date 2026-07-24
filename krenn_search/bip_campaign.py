import sys, time, json
import numpy as np
sys.path.insert(0, sys.path[0] or ".")
from bip_scan import gen_support
from bipartite_fast import FastBipartite

m, d, hours, seed, tag = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rng = np.random.default_rng(seed)
t_end = time.time() + hours * 3600
best = np.inf
r = 0
while time.time() < t_end:
    supp = gen_support(m, d, rng)
    if supp is None:
        continue
    s = FastBipartite(m, d, supp)
    cost, w = s.solve(restarts=2, seed=int(rng.integers(1e9)), max_nfev=300)
    if cost < best:
        best = cost
        with open(f"{tag}_best.json", "w") as fh:
            json.dump({"cost": cost, "support": {str(k): v for k, v in supp.items()},
                       "w_re": list(np.real(w)), "w_im": list(np.imag(w))}, fh)
    print(f"[{tag}] r={r} active={s.n_active()} cost={cost:.4e} best={best:.4e}", flush=True)
    if best < 1e-20:
        print(f"*** [{tag}] SOLUTION ***", flush=True)
        break
    r += 1
