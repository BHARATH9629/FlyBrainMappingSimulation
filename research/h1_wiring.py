"""H1 wiring measures (pre-registration, section 4), computed from the wiring only -- never from attack results.

  R1  strong drivers: distinct neurons with >= 10 excitatory synapses onto the readout(s) that fire in the healthy
      stimulus run (seed 1).
  R2  vertex cut: the smallest number of neurons whose removal disconnects every excitatory path (edges of >= 5
      synapses, at most 4 steps) from the stimulus neurons to the readout(s). Implementation: keep only neurons that
      lie on some stimulus -> readout path of <= 4 steps (distance from the stimulus + distance to the readout <= 4),
      split every neuron into in/out with capacity 1 (the readouts themselves cannot be cut, as for the attacker), and
      take the maximum flow = minimum vertex cut (Menger's theorem) on that sub-graph.

Usage: python research/h1_wiring.py [instance ...]        (FlyWire)
       python research/h1_wiring.py --banc [instance ...]  (BANC, scaled weights; synapse counts are what matter here)
"""
import json, os, sys
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from guard import start, finish
out = start("h1")
import instances
from behaviours import Behaviour

MIN_SYN_R1, MIN_SYN_R2, MAX_STEPS = 10, 5, 4
BANC = "--banc" in sys.argv; args = [a for a in sys.argv[1:] if not a.startswith("--")]
SCREEN = sys.argv[sys.argv.index("--screen") + 1] if "--screen" in sys.argv else None   # FlyWire / BANC / male
if SCREEN:                                             # the screen attack set ("s_" instances) in that fly's primary model
    args = [a for a in args if a != SCREEN]; SCF = os.path.join(os.path.dirname(out), "{}", "scale_factor.json")
    if SCREEN == "BANC":
        from banc import BancBrain; brain = BancBrain(scale=json.load(open(SCF.format("banc")))["median_ratio"])
    elif SCREEN == "male":
        from malecns import MaleBrain; brain = MaleBrain(scale=json.load(open(SCF.format("male")))["median_ratio"])
    else:
        from brain import Brain; brain = Brain("female")
    names = args or instances.register_screen(brain, SCREEN); instances.register_screen(brain, SCREEN)
elif BANC:
    from banc import BancBrain
    brain = BancBrain(scale=json.load(open(os.path.join(os.path.dirname(out), "banc", "scale_factor.json")))["median_ratio"])
    names = args or instances.register_banc(); instances.register_banc()
else:
    from brain import Brain
    brain = Brain("female"); instances.register(brain=brain)
    names = args or ["eat", "drink", "escape", "retreat", "turn", "r_taste_L", "r_escape_R"]
synapses = (brain.W / getattr(brain, "scale", 1.0))                       # back to raw synapse counts
E = synapses.multiply(synapses >= MIN_SYN_R2).tocsr(); E.eliminate_zeros()      # excitatory, >= 5 synapses
E.data[:] = 1


def hops(A, seeds, maxd):
    """breadth-first distance (<= maxd) from a set of seeds along A's edges"""
    d = np.full(A.shape[0], np.inf); d[list(seeds)] = 0; frontier = np.array(list(seeds))
    for k in range(1, maxd + 1):
        nxt = np.unique(A[frontier].indices); nxt = nxt[np.isinf(d[nxt])]
        if nxt.size == 0: break
        d[nxt] = k; frontier = nxt
    return d


def vertex_cut(stim, readouts):
    ds = hops(E, stim, MAX_STEPS); dt = hops(E.T.tocsr(), readouts, MAX_STEPS)
    keep = np.flatnonzero(ds + dt <= MAX_STEPS)
    if not set(readouts) & set(keep.tolist()): return 0, int(keep.size)
    loc = {int(g): i for i, g in enumerate(keep)}; n = keep.size
    sub = E[keep][:, keep].tocoo()
    # node i -> (in = i, out = n + i); source = 2n, sink = 2n + 1
    BIG = 10 ** 6; ro = {loc[r] for r in readouts if r in loc}
    rows, cols, cap = [], [], []
    for i in range(n): rows.append(i); cols.append(n + i); cap.append(BIG if i in ro else 1)       # cut a neuron = 1
    rows += list(n + sub.row); cols += list(sub.col); cap += [BIG] * sub.nnz                     # wires are not cut
    for s in stim:
        if int(s) in loc: rows.append(2 * n); cols.append(loc[int(s)]); cap.append(BIG)
    for r in ro: rows.append(n + r); cols.append(2 * n + 1); cap.append(BIG)
    G = sp.csr_matrix((np.array(cap, np.int32), (rows, cols)), shape=(2 * n + 2, 2 * n + 2))
    return int(maximum_flow(G, 2 * n, 2 * n + 1).flow_value), int(n)


res = {}
for name in names:
    b = Behaviour(brain, name)
    r = brain.run(b.stim, record=b.readouts, seed=1, **b.sim); active = set(r["active"])
    col = np.asarray(synapses[:, b.readouts].sum(1)).ravel()
    strong = [int(i) for i in np.flatnonzero(col >= MIN_SYN_R1) if int(i) in active and int(i) not in b.readouts]
    cut, n_sub = vertex_cut([int(s) for s in b.stim], [int(x) for x in b.readouts])
    res[name] = dict(R1_strong_drivers=len(strong), R2_vertex_cut=cut, R2_subgraph_neurons=n_sub,
                     n_stimulus=len(b.stim), readouts=len(b.readouts))
    print(f"{name:12s} R1 strong drivers {len(strong):3d} | R2 vertex cut {cut:3d} (sub-graph {n_sub:,} neurons)", flush=True)
fn = os.path.join(out, f"wiring_screen_{SCREEN}.json" if SCREEN else "wiring_banc.json" if BANC else "wiring_flywire.json")
old = json.load(open(fn)) if os.path.exists(fn) else {}
json.dump(old | res, open(fn, "w"), indent=1)
finish()
