"""NeuroRedTeam v2: robust attack + repair for one behaviour, every goal verified on 3 seeds.
Usage:  python run_redteam.py [--behaviour eat] [--goal block|force|both] [--sex female|male] [--quiet]"""
import argparse, json, time, os
import numpy as np
from brain import Brain
from behaviours import Behaviour, RULES, BEHAVIOURS
from agents import Attacker, Defender, Log, redundancy, CHECK_SEEDS
FRESH_SEEDS = (11, 12, 13)                   # held out of every search, used only to confirm a finished attack

ap = argparse.ArgumentParser()
ap.add_argument("--behaviour", default="eat"); ap.add_argument("--goal", default="both", choices=["block", "force", "both", "corrupt"])
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--sex", default="female", choices=["female", "male"])
ap.add_argument("--side", default="default", help="stimulus side override: L, R or both")
ap.add_argument("--no_search", action="store_true", help="skip the beam-search attacker (slow on the male)")
args = ap.parse_args()

t0 = time.time()
brain = Brain(args.sex); print(f"brain loaded ({args.sex} fly): {brain.N:,} neurons, {brain.W.nnz:,} connections")
b = Behaviour(brain, args.behaviour, side=(None if args.side == "both" else args.side)); log = Log(quiet=args.quiet)

print(f"\n== Healthy baseline for '{b.name}' (seeds {CHECK_SEEDS}) ==")
ons = [b.measure(True, seed=s)[0] for s in CHECK_SEEDS]; offs = [b.measure(False, seed=s)[0] for s in CHECK_SEEDS]
print(f"  with stimulus {np.mean(ons):.1f} +/- {np.std(ons):.1f} Hz, without {np.mean(offs):.1f} Hz   "
      f"(sub-network {b.measure(True)[1]['n_sub']:,} neurons, ~{(time.time()-t0)/6:.1f}s per run)")

entry = {"behaviour": b.name, "healthy_on_hz": float(np.mean(ons)), "healthy_off_hz": float(np.mean(offs)), "goals": {}}
goals = ["block", "force"] if args.goal == "both" else [args.goal]
if b.expected == "off" and args.goal == "both": goals = ["corrupt"]     # a refused stimulus: the only sensible goal
for goal in goals:
    what = {"block": "make the behaviour disappear with the stimulus present", "force": "trigger the behaviour with NO stimulus",
            "corrupt": "make a refused stimulus trigger the behaviour"}[goal]
    print(f"\n== RED TEAM, goal = {goal.upper()} ({what}) ==")
    results = {}
    for strat in (("greedy",) if goal == "corrupt" else ("greedy", "inject_inhibitors", "weaken_only")):
        a = Attacker(b, log, goal=goal)
        res = getattr(a, strat)()
        results[strat] = res
        print(f"  {strat:18s}: " + (f"size {res['size']}  ({res['simulations']} sims)  {res['intervention']}" if res else "no attack found within budget"))
    found = [r for r in results.values() if r]
    best = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
    if not args.no_search:                                   # can beam search beat the greedy strategies?
        cap = (best["size"] - 1) if best else RULES["moves"]["lesion"]["max"]
        srch = Attacker(b, log, goal=goal).search(max_size=max(cap, 1)) if cap >= 1 else None
        results["search"] = srch
        print(f"  {'search':18s}: " + (f"size {srch['size']}  ({srch['simulations']} sims)  {srch['intervention']}" if srch
                                       else f"nothing smaller than {best['size']} within budget" if best else "no attack found within budget"))
        if srch and (best is None or srch["size"] < best["size"]): best = srch
    # every attack is re-checked on seeds the search never used: an attack that only works on its own seeds is
    # marginal, not a result.
    if best:
        fresh = [b.measure(goal in ("block", "corrupt"), seed=s, **best["intervention"])[0] for s in FRESH_SEEDS]
        holds = all(r <= b.off_thr for r in fresh) if goal == "block" else all(r >= b.on_thr for r in fresh)
        best["fresh_seeds"] = [round(r, 1) for r in fresh]; best["verified_on_unseen_seeds"] = bool(holds)
        print(f"  verify on unseen seeds {FRESH_SEEDS}: {[round(r) for r in fresh]} Hz -> "
              + ("holds" if holds else "DOES NOT HOLD (marginal attack, reported as such)"))
    g = {"attacks": results, "best": best, "greedy_best_size": min((r["size"] for k, r in results.items() if r and k != "search"), default=None)}

    if goal in ("block", "corrupt") and best:
        print(f"\n  redundancy: how many DISJOINT lesion attacks exist?")
        sets = redundancy(b, log, goal=goal)
        g["disjoint_attacks"] = [s["size"] for s in sets]
        print(f"  -> {len(sets)} disjoint attacks of sizes {g['disjoint_attacks']}")
    if goal == "corrupt" and best:
        ctrl_name = (b.cfg.get("control") if hasattr(b, "cfg") else None) or BEHAVIOURS[args.sex][b.name].get("control")
        ctrl = Behaviour(brain, ctrl_name) if ctrl_name else None
        print("\n== BLUE TEAM: restore the refusal after the CORRUPT attack " + f"({best['strategy']}, size {best['size']}) ==")
        if ctrl: print(f"  the repair must also keep '{ctrl_name}' working (>= {ctrl.on_thr:.0f} Hz): no fixing this by breaking feeding")
        rep = Defender(b, best, log).restore_off(control=ctrl)
        g["repair"] = rep
        if rep and rep.get("partial"): print(f"  -> BLUE TEAM LOST: {rep.get('why', '')}")
        else: print("  -> " + (f"REPAIR: {rep['move']} on {rep['size']} neurons; refused stimulus back to {rep['on_hz']:.1f} Hz"
                         + (f", '{ctrl_name}' still {rep['control_hz']:.1f} Hz" if ctrl else "") + f"  ({rep['simulations']} sims)"
                         if rep else "no repair found within budget"))
    if goal == "block" and best:
        print(f"\n== BLUE TEAM: repair after the best BLOCK attack ({best['strategy']}, size {best['size']}) ==")
        # restore(): back to >= 80 % of healthy on every seed. greedy() (fewest neurons that clear the bar) is recorded
        # alongside for comparison.
        rep = Defender(b, best, log).restore(target=0.8)
        old = Defender(b, best, log).greedy()
        g["repair_greedy"] = old
        # no fallback to greedy: restore() itself returns its strongest working repair when 80 % is out of reach
        g["repair"] = rep
        if rep and not rep.get("partial"):
            rec = f", worst run {rep['recovery'] * 100:.0f}% of healthy" if "recovery" in rep else ""
            print(f"  -> REPAIR: {rep['move']} on {rep['size']} neurons; {rep['on_hz']:.1f} Hz with the stimulus, "
                  f"{rep['off_hz']:.1f} Hz without{rec}  ({rep['simulations']} sims)")
            if old: print(f"     (old greedy defender: {old['size']} neurons, {old['on_hz']:.1f} Hz)")
        else: print("  -> no repair found within budget (atlas entry: hard to repair)")
    entry["goals"][goal] = g

entry["total_seconds"] = round(time.time() - t0)
out_dir = f"atlas/{args.sex}"; os.makedirs(out_dir, exist_ok=True)
def _jsonable(o):
    if isinstance(o, dict): return {(str(k) if isinstance(k, tuple) else k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    return o
entry["sex"] = args.sex; entry["side"] = args.side
tag = "" if args.side == "default" else f"_side{args.side}"
json.dump(_jsonable({"entry": entry, "log": log.rows}), open(f"{out_dir}/{b.name}{tag}.json", "w"), indent=1, default=int)

print(f"\n== FRAGILITY ATLAS: {b.name}, {args.sex} fly  (total {entry['total_seconds']}s, all goals verified on {len(CHECK_SEEDS)} seeds) ==")
for goal, g in entry["goals"].items():
    bst = g["best"]
    line = f"  {goal:5s}: " + (f"smallest attack = {bst['size']} neurons via {bst['strategy']}" if bst else "no attack found")
    if "disjoint_attacks" in g: line += f" | disjoint attacks: {g['disjoint_attacks']}"
    if g.get("repair") and g["repair"].get("partial"):
        line += " | repair: BLUE TEAM LOST"
    elif g.get("repair"):
        rp = g["repair"]; line += f" | repair = {rp['size']} neurons"
        if entry["healthy_on_hz"] > 0:                       # a behaviour that should be ON: how much came back
            line += f" ({rp['on_hz']/entry['healthy_on_hz']*100:.0f} % recovery)"
        else:                                                # a behaviour that should be OFF (refuse_bitter): how quiet it is again
            line += f" (refused stimulus back to {rp['on_hz']:.1f} Hz" + (f", control {rp['control_hz']:.0f} Hz)" if rp.get("control_hz") is not None else ")")
    print(line)
print(f"  saved {out_dir}/{b.name}{tag}.json")
