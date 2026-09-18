"""NeuroRedTeam v2: robust attack + repair for one behaviour, every goal verified on 3 seeds.
Usage:  python run_redteam.py [--behaviour eat] [--goal block|force|both] [--quiet]"""
import argparse, json, time, os
import numpy as np
from brain import Brain
from behaviours import Behaviour
from agents import Attacker, Defender, Log, redundancy, CHECK_SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--behaviour", default="eat"); ap.add_argument("--goal", default="both", choices=["block", "force", "both"])
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

t0 = time.time()
brain = Brain(); print(f"brain loaded: {brain.N:,} neurons, {brain.W.nnz:,} synapses")
b = Behaviour(brain, args.behaviour); log = Log(quiet=args.quiet)

print(f"\n== Healthy baseline for '{b.name}' (seeds {CHECK_SEEDS}) ==")
ons = [b.measure(True, seed=s)[0] for s in CHECK_SEEDS]; offs = [b.measure(False, seed=s)[0] for s in CHECK_SEEDS]
print(f"  with stimulus {np.mean(ons):.1f} +/- {np.std(ons):.1f} Hz, without {np.mean(offs):.1f} Hz   "
      f"(sub-network {b.measure(True)[1]['n_sub']:,} neurons, ~{(time.time()-t0)/6:.1f}s per run)")

entry = {"behaviour": b.name, "healthy_on_hz": float(np.mean(ons)), "healthy_off_hz": float(np.mean(offs)), "goals": {}}
goals = ["block", "force"] if args.goal == "both" else [args.goal]
for goal in goals:
    print(f"\n== RED TEAM, goal = {goal.upper()} "
          f"({'make the fly stop eating with sugar present' if goal == 'block' else 'make the fly eat with NO sugar'}) ==")
    atk = Attacker(b, log, goal=goal)
    results = {}
    for strat in ("greedy", "inject_inhibitors", "weaken_only"):
        a = Attacker(b, log, goal=goal)
        res = getattr(a, strat)()
        results[strat] = res
        print(f"  {strat:18s}: " + (f"size {res['size']}  ({res['simulations']} sims)  {res['intervention']}" if res else "no attack found within budget"))
    found = [r for r in results.values() if r]
    best = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
    g = {"attacks": results, "best": best}

    if goal == "block" and best:
        print(f"\n  redundancy: how many DISJOINT lesion attacks exist?")
        sets = redundancy(b, log, goal="block")
        g["disjoint_attacks"] = [s["size"] for s in sets]
        print(f"  -> {len(sets)} disjoint attacks of sizes {g['disjoint_attacks']}")

        print(f"\n== BLUE TEAM: repair after the best BLOCK attack ({best['strategy']}, size {best['size']}) ==")
        rep = Defender(b, best, log).greedy()
        g["repair"] = rep
        print("  -> " + (f"REPAIR: {rep['move']} on {rep['size']} neurons; {rep['on_hz']:.1f} Hz with sugar, {rep['off_hz']:.1f} Hz without  ({rep['simulations']} sims)"
                         if rep else "no repair found within budget (atlas entry: hard to repair)"))
    entry["goals"][goal] = g

entry["total_seconds"] = round(time.time() - t0)
os.makedirs("atlas", exist_ok=True)
def _jsonable(o):
    if isinstance(o, dict): return {(str(k) if isinstance(k, tuple) else k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    return o
json.dump(_jsonable({"entry": entry, "log": log.rows}), open(f"atlas/{b.name}.json", "w"), indent=1, default=int)

print(f"\n== FRAGILITY ATLAS: {b.name}  (total {entry['total_seconds']}s, all goals verified on {len(CHECK_SEEDS)} seeds) ==")
for goal, g in entry["goals"].items():
    bst = g["best"]
    line = f"  {goal:5s}: " + (f"smallest attack = {bst['size']} neurons via {bst['strategy']}" if bst else "no attack found")
    if "disjoint_attacks" in g: line += f" | disjoint attacks: {g['disjoint_attacks']}"
    if g.get("repair"): line += f" | repair = {g['repair']['size']} neurons ({g['repair']['on_hz']/entry['healthy_on_hz']*100:.0f} % recovery)"
    print(line)
print(f"  saved atlas/{b.name}.json")
