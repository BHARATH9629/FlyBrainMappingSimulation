"""H5: are the weak spots the same across the two hemispheres and across two flies? (pre-registration, section 7)

  overlap  : Jaccard index of the cell types in two matched attacks vs 10,000 random draws of the same numbers of
             cell types from each output's direct + second-step input neurons.
  transfer : move an attack to the other side / other fly by cell type (all neurons of each attacked type on the target
             side, same move and strength) and test it on the 15 unseen seeds; "breaks" = goal met on >= 12 of 15.

Usage: python research/h5_reproducibility.py
"""
import json, os
import numpy as np
from guard import start, finish
out = start("h5")
import instances
from brain import Brain
from banc import BancBrain
from behaviours import Behaviour

FRESH = tuple(range(11, 26)); VERIFY_STRATEGIES = ("greedy", "inject_inhibitors", "weaken_only")
fw = Brain("female"); instances.register(brain=fw)
bb = BancBrain(scale=json.load(open(os.path.join(os.path.dirname(out), "banc", "scale_factor.json")))["median_ratio"])
instances.register_banc()
BRAIN = {"FlyWire": fw, "BANC": bb}


def discovery_attack(name, goal):
    """the attack the Phase 3 verification used: smallest over greedy / inject_inhibitors / weaken_only"""
    atks = json.load(open(os.path.join(instances.guard.ROOT, "neuroredteam", "atlas", "female", f"{name}.json")))["entry"]["goals"][goal]["attacks"]
    c = [a for k, a in atks.items() if a and k in VERIFY_STRATEGIES]
    return min(c, key=lambda a: a["size"])["intervention"]


def research_attack(fn):
    return json.load(open(os.path.join(os.path.dirname(out), "instances", f"{fn}.json")))["attack"]["intervention"]


CASES = {  # label: (dataset, behaviour name, goal, intervention)
    "escape FlyWire L (discovery)": ("FlyWire", "escape", "block", discovery_attack("escape", "block")),
    "escape FlyWire R":             ("FlyWire", "r_escape_R", "block", research_attack("r_escape_R_block")),
    "turn FlyWire R-eye (discovery)": ("FlyWire", "turn", "block", discovery_attack("turn", "block")),
    "turn BANC R-eye":              ("BANC", "b_turn_R", "block", research_attack("b_turn_R_block")),
    "turn BANC L-eye":              ("BANC", "b_turn_L", "block", research_attack("b_turn_L_block")),
    "bitter FlyWire L (discovery)": ("FlyWire", "refuse_bitter", "corrupt", discovery_attack("refuse_bitter", "corrupt")),
    "bitter FlyWire R":             ("FlyWire", "r_bitter_R", "corrupt", research_attack("r_bitter_R_corrupt")),
}
PAIRS = [("escape FlyWire L (discovery)", "escape FlyWire R", "mirror"),
         ("turn FlyWire R-eye (discovery)", "turn BANC R-eye", "same side"),
         ("turn BANC R-eye", "turn BANC L-eye", "mirror"),
         ("bitter FlyWire L (discovery)", "bitter FlyWire R", "mirror")]


def neurons(iv):
    return [(k, int(i), v) for k in ("inject", "lesion", "weaken") for i, v in
            ((iv.get(k) or {}).items() if isinstance(iv.get(k), dict) else [(x, None) for x in (iv.get(k) or [])])]


def ctypes(brain, iv): return {brain.type_of(i) for _, i, _ in neurons(iv) if brain.type_of(i)}


def near_types(brain, name):
    """cell types of the output's direct + second-step input neurons (the null pool)"""
    b = Behaviour(brain, name); W = brain.W.tocsc(); first = set()
    for r in b.readouts: first |= set(W[:, r].indices.tolist())
    second = set()
    for j in first: second |= set(W[:, j].indices.tolist())
    return sorted({brain.type_of(i) for i in (first | second) if brain.type_of(i)})


def jaccard(a, b): return len(a & b) / len(a | b) if (a | b) else 0.0


def transfer(src, dst, relation):
    ds_s, _, _, iv = CASES[src]; ds_t, name_t, goal, _ = CASES[dst]; bs, bt = BRAIN[ds_s], BRAIN[ds_t]
    T = np.array([str(x) for x in bt.types["type"]]); S = bt.types["side"].astype(str).values
    new = {"inject": {}, "lesion": [], "weaken": {}}; dropped = set()
    for k, i, v in neurons(iv):
        ty = bs.type_of(i); side = str(bs.types["side"].iloc[i])
        if relation == "mirror": side = {"L": "R", "R": "L"}.get(side, side)
        tgt = np.flatnonzero((T == ty) & (S == side)) if ty else []
        if len(tgt) == 0: dropped.add(ty); continue
        for j in tgt:
            if k == "lesion": new["lesion"].append(int(j))
            else: new[k][int(j)] = v
    new = {k: v for k, v in new.items() if v}
    b = Behaviour(bt, name_t)
    r = [b.measure(goal in ("block", "corrupt"), seed=s, **new)[0] for s in FRESH]
    ok = [x <= b.off_thr for x in r] if goal == "block" else [x >= b.on_thr for x in r]
    return dict(n_neurons=sum(len(v) for v in new.values()), dropped_types=sorted(dropped), goal_met=int(sum(ok)),
                breaks=bool(sum(ok) >= 12), mean_hz=float(np.mean(r)))


rng = np.random.default_rng(0); res = {"pairs": []}
for a, b, rel in PAIRS:
    (da, na, _, iva), (db, nb, _, ivb) = CASES[a][:4], CASES[b][:4]
    ta, tb = ctypes(BRAIN[da], iva), ctypes(BRAIN[db], ivb); J = jaccard(ta, tb)
    pa, pb = near_types(BRAIN[da], na), near_types(BRAIN[db], nb)
    null = [jaccard(set(rng.choice(pa, min(len(ta), len(pa)), replace=False)), set(rng.choice(pb, min(len(tb), len(pb)), replace=False)))
            for _ in range(10_000)]
    p = float((np.sum(np.array(null) >= J) + 1) / (len(null) + 1))
    t1, t2 = transfer(a, b, rel), transfer(b, a, rel)
    row = dict(a=a, b=b, relation=rel, types_a=sorted(ta), types_b=sorted(tb), shared=sorted(ta & tb), jaccard=J,
               null_mean=float(np.mean(null)), p=p, pool_a=len(pa), pool_b=len(pb), transfer_a_to_b=t1, transfer_b_to_a=t2)
    res["pairs"].append(row)
    print(f"{a}  <->  {b}\n   shared types {sorted(ta & tb)}\n   Jaccard {J:.2f} vs chance {np.mean(null):.3f} (pool {len(pa)}/{len(pb)} types), p = {p:.4f}"
          f"\n   transfer ->: {t1['goal_met']}/15 {'BREAKS' if t1['breaks'] else 'does not break'} ({t1['n_neurons']} neurons, dropped {t1['dropped_types']})"
          f"\n   transfer <-: {t2['goal_met']}/15 {'BREAKS' if t2['breaks'] else 'does not break'} ({t2['n_neurons']} neurons, dropped {t2['dropped_types']})", flush=True)
json.dump(res, open(os.path.join(out, "h5.json"), "w"), indent=1)
finish()
