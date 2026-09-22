"""Null models: is the red team's attack better than obvious baselines of the same size?

For a behaviour whose minimal block attack has k lesions, we ask how often k lesions chosen WITHOUT simulation also
block it: (a) k random neurons from the active sub-network, (b) the k highest-degree active neurons, (c) the k
strongest excitatory inputs to the readout, (d) the k most active neurons. Each candidate set is judged by exactly the
same referee as the attacker (Attacker._check: goal met on seeds 1-3).
Usage: python nullmodels.py --sex female --behaviour eat --reps 200
"""
import argparse, json, os
import numpy as np
from brain import Brain
from behaviours import Behaviour
from agents import Attacker, Log

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--goal", default="block", choices=["block", "force", "corrupt"])
ap.add_argument("--k", type=int, default=0, help="attack size to match (0 = read it from the atlas)")
ap.add_argument("--move", default="lesion", choices=["lesion", "inject"], help="the move the red team used for this goal")
ap.add_argument("--reps", type=int, default=200, help="random sets to draw"); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas", args.sex)

brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); W = brain.W
atk = Attacker(b, Log(quiet=True), goal=args.goal)

k = args.k
if not k:                                                   # the size the red team achieved, from the atlas
    e = json.load(open(os.path.join(out_dir, f"{args.behaviour}.json")))["entry"]["goals"][args.goal]["attacks"]
    sizes = [len(a["intervention"][args.move]) for a in e.values() if a and a.get("intervention", {}).get(args.move)]
    if not sizes: raise SystemExit(f"no {args.move}-only {args.goal} attack in atlas/{args.sex}/{args.behaviour}.json -- try --move inject")
    k = min(sizes); print(f"attacker's smallest {args.move}-only {args.goal}: {k} neurons")

_, r0 = b.measure(True)
active = {int(i): hz for i, hz in r0["active"].items() if i not in b.protected}
pool = sorted(active)
col = W[:, b.readout].toarray().ravel()
deg = np.asarray((abs(W) > 0).sum(axis=1)).ravel() + np.asarray((abs(W) > 0).sum(axis=0)).ravel()
HZ = 150.0
def iv_of(sel): return {"lesion": list(sel)} if args.move == "lesion" else {"inject": {int(i): HZ for i in sel}}
print(f"{args.sex} {b.name}: {len(pool)} active neurons; testing {args.move} sets of {k} against the same referee", flush=True)

results = {}
top = lambda score: [int(i) for i in sorted(pool, key=lambda j: -score[j])[:k]]
for label, s in (("highest degree", deg), ("strongest input to readout", np.abs(col)),
                 ("strongest inhibitor of readout", np.maximum(-col, 0)),
                 ("most active", np.array([active.get(i, 0.0) for i in range(brain.N)]))):
    les = top(s)
    ok, hz = atk._check(iv_of(les))
    results[label] = {"set": les, "types": [brain.type_of(i) for i in les], "hz": round(hz, 1), "blocked": bool(ok)}
    print(f"  {label:28s} -> {hz:6.1f} Hz  {'GOAL MET' if ok else 'no'}", flush=True)

hits, rates = 0, []
for n in range(args.reps):
    les = [int(x) for x in rng.choice(pool, k, replace=False)]
    ok, hz = atk._check(iv_of(les)); hits += ok; rates.append(hz)
    if n % 25 == 0: print(f"  random {n}/{args.reps}: {hits} hits so far", flush=True)
print(f"\n  random {k}-neuron {args.move} sets: {hits}/{args.reps} met the goal ({100*hits/args.reps:.1f} %), "
      f"mean readout {np.mean(rates):.1f} Hz (healthy {r0['rates'][b.readout]:.0f} Hz)")
results["random"] = {"move": args.move, "reps": args.reps, "hits": int(hits), "rate": hits / args.reps, "mean_hz": round(float(np.mean(rates)), 1),
                     "ci95_upper_if_zero": round(1 - 0.05 ** (1 / args.reps), 4) if hits == 0 else None}
json.dump({"sex": args.sex, "behaviour": b.name, "goal": args.goal, "move": args.move, "k": k, "healthy_hz": r0["rates"][b.readout],
           "baselines": results}, open(os.path.join(out_dir, f"nullmodels_{b.name}_{args.goal}.json"), "w"), indent=1)
print(f"saved atlas/{args.sex}/nullmodels_{b.name}_{args.goal}.json")
