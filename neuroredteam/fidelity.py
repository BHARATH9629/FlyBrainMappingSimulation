"""Fidelity check: does the fast sub-network give the same MN9 rates as the full brain?

For the healthy brain and for the best BLOCK attack from the atlas, run the stimulus on 3 seeds at several
sub-network sizes up to the whole brain, and print the readout rate. If the numbers agree, the atlas numbers can be
trusted; if not, raise depth / lower min_syn in Brain.run.
Usage: python fidelity.py --sex male
"""
import argparse, json, os, time
import numpy as np
from brain import Brain
from behaviours import Behaviour
from agents import CHECK_SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="male", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
args = ap.parse_args()


def _iv(d):
    out = {}
    for k, v in (d or {}).items():
        if k == "lesion": out[k] = [int(x) for x in v]
        elif k == "potentiate": out[k] = {tuple(int(t) for t in key.strip("()").split(",")): float(f) for key, f in v.items()}
        else: out[k] = {int(a): float(b) for a, b in v.items()}
    return out


brain = Brain(args.sex); b = Behaviour(brain, args.behaviour)
entry = json.load(open(f"atlas/{args.sex}/{args.behaviour}.json"))["entry"]
# conditions: healthy, the best attack of every goal that has one (block/corrupt with stimulus, force without), repair
conds = {"healthy": ({}, True)}
for goal, g in entry["goals"].items():
    best = g.get("best")
    if not best: print(f"  (no {goal} attack in the atlas entry: skipped)"); continue
    attack = _iv(best["intervention"]); with_stim = goal != "force"
    conds[f"{goal} ({best['strategy'].replace('attacker-', '')}, {best['size']})"] = (attack, with_stim)
    rep = g.get("repair")
    if rep: conds[f"repaired (+{rep['size']})"] = ({**attack, **_iv(rep["intervention"])}, True)
levels = [("depth 3, >=5 syn (atlas default)", dict(depth=3, min_syn=5)), ("depth 4, >=5 syn", dict(depth=4, min_syn=5)),
          ("depth 3, >=1 syn", dict(depth=3, min_syn=1)), ("FULL BRAIN", dict(depth=99, min_syn=1))]

rows = []
print(f"{args.sex} fly, behaviour {b.name}: readout rate (Hz) with stimulus, mean +/- sd over seeds {CHECK_SEEDS}\n")
print(f"{'sub-network':34s} {'neurons':>9s}  " + "  ".join(f"{c[:22]:>22s}" for c in conds) + "   s/run")
for name, kw in levels:
    line = f"{name:34s}"; n_sub = None; t0 = time.time(); nruns = 0; res = {}
    for cname, (iv, with_stim) in conds.items():
        rates = []
        for s in CHECK_SEEDS:
            rate, r = b.measure(with_stim, seed=s, **iv, **kw); rates.append(rate); n_sub = r["n_sub"]; nruns += 1
        res[cname] = (float(np.mean(rates)), float(np.std(rates)))
    line += f" {n_sub:9,d}  " + "  ".join(f"{m:12.1f} +/- {sd:4.1f}" for m, sd in res.values()) + f"   {(time.time()-t0)/nruns:5.1f}"
    print(line, flush=True)
    rows.append(dict(level=name, n_sub=n_sub, **{c: res[c] for c in conds}))
os.makedirs(f"atlas/{args.sex}", exist_ok=True)
json.dump(rows, open(f"atlas/{args.sex}/fidelity_{b.name}.json", "w"), indent=1)
print(f"\nsaved atlas/{args.sex}/fidelity_{b.name}.json")
print("verdict: the atlas default is trustworthy if its numbers stay on the same side of the 40 / 10 Hz thresholds as FULL BRAIN.")
