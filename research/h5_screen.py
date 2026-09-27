"""H5 overlap on the screen data (pre-registration log, 27 Sep): do matched attacks in different flies use the same cell
types? Types matched by FlyWire name (as in the screen). Null: same number of types drawn uniformly from the types within
two synaptic steps upstream of that fly's readout (any sign, >= 5 synapses per step). Statistic: mean Jaccard over all
fly pairs; one-sided p from 10,000 null means. Primary: pre-registered scales; secondary: BANC x1.0."""
import glob, json, os, itertools
import numpy as np
from guard import start, finish
out = start("h5")
import instances
from brain import Brain
from banc import BancBrain
from malecns import MaleBrain
import sys
STRICT = "--strict" in sys.argv          # exploratory stricter null (27 Sep): direct inputs only
R = os.path.dirname(out); SCF = os.path.join(R, "{}", "scale_factor.json")
brains = {"FlyWire": Brain("female"), "BANC": BancBrain(scale=json.load(open(SCF.format("banc")))["median_ratio"]),
          "male": MaleBrain(scale=json.load(open(SCF.format("male")))["median_ratio"])}
BAD = {"nan", "None", ""}
info = {}
for f, b in brains.items():
    instances.register_screen(b, f); names = instances.screen_names(b, f)
    A = (abs(b.W / getattr(b, "scale", 1.0)) >= 5).astype(np.int8).tocsc()
    info[f] = dict(names=names, A=A, brain=b)


def attack_types(f, iv):
    idx = set(int(i) for i in iv.get("lesion", [])) | {int(k) for k in ("inject", "weaken", "bias") for k in [] }
    for k in ("inject", "weaken", "bias"): idx |= {int(i) for i in iv.get(k, {})}
    return {info[f]["names"][i] for i in idx} - BAD


def pool(f, name):
    from behaviours import Behaviour
    b = Behaviour(info[f]["brain"], name); A = info[f]["A"]
    one = set(np.unique(A[:, b.readouts].indices).tolist())
    if STRICT: return sorted({info[f]["names"][i] for i in one - set(b.readouts)} - BAD)
    two = set(np.unique(A[:, sorted(one)].indices).tolist()) if one else set()
    return sorted({info[f]["names"][i] for i in (one | two) - set(b.readouts)} - BAD)


SETS = {"primary": {"FlyWire": "instances/s_{k:02d}_FlyWire_block.json", "BANC": "instances/s_{k:02d}_BANC_block.json",
                    "male": "instances/s_{k:02d}_male_block.json"},
        "secondary (BANC x1.0)": {"FlyWire": "instances/s_{k:02d}_FlyWire_block.json", "BANC": "sweep_BANC_x1.0/s_{k:02d}_BANC_block.json",
                                  "male": "instances/s_{k:02d}_male_block.json"}}
rng = np.random.default_rng(0); N = 10000; rep = {}; pools = {}
for label, pats in SETS.items():
    pairs = []
    for k in range(35):
        at = {}
        for f, p in pats.items():
            fn = os.path.join(R, p.format(k=k))
            if not os.path.exists(fn): continue
            r = json.load(open(fn))
            if r.get("included", True) and r.get("attack"): at[f] = attack_types(f, r["attack"]["intervention"])
        for f1, f2 in itertools.combinations(sorted(at), 2):
            if not at[f1] or not at[f2]: continue
            for f in (f1, f2):
                if (f, k) not in pools: pools[(f, k)] = pool(f, f"s_{k:02d}_{f}")
            j = len(at[f1] & at[f2]) / len(at[f1] | at[f2])
            pairs.append(dict(group=k, flies=[f1, f2], types=[sorted(at[f1]), sorted(at[f2])], shared=sorted(at[f1] & at[f2]), jaccard=j,
                              pool_sizes=[len(pools[(f1, k)]), len(pools[(f2, k)])]))
    null = np.zeros(N)
    for p in pairs:
        (f1, f2), k = p["flies"], p["group"]; P1, P2 = pools[(f1, k)], pools[(f2, k)]; n1, n2 = len(p["types"][0]), len(p["types"][1])
        for t in range(N):
            a = set(rng.choice(P1, min(n1, len(P1)), replace=False)); b = set(rng.choice(P2, min(n2, len(P2)), replace=False))
            null[t] += len(a & b) / len(a | b)
    null /= len(pairs); obs = float(np.mean([p["jaccard"] for p in pairs])); pv = (np.sum(null >= obs) + 1) / (N + 1)
    rep[label] = dict(pairs=pairs, mean_jaccard=obs, null_mean=float(null.mean()), p_one_sided=float(pv), confirmed=bool(pv < 0.05),
                      pairs_with_any_shared=sum(bool(p["shared"]) for p in pairs))
    print(f"[{label}] {len(pairs)} fly pairs: mean Jaccard {obs:.3f} vs null {null.mean():.3f}, p = {pv:.4f}; "
          f"{rep[label]['pairs_with_any_shared']} pairs share at least one type  -> overlap {'CONFIRMED' if pv < 0.05 else 'NOT confirmed'}", flush=True)
    for p in pairs:
        if p["shared"]: print(f"    group {p['group']:2d} {p['flies']}: shared {p['shared']}  (J = {p['jaccard']:.2f})")
json.dump(rep, open(os.path.join(out, "h5_screen_strict.json" if STRICT else "h5_screen.json"), "w"), indent=1)
finish()
