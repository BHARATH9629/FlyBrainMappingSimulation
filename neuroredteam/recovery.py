"""Recovery curve: how big a repair does it take to restore the behaviour to a given fraction of healthy?

For the atlas's best BLOCK attack, raise the ON bar to 40, 60, 80, 100 % of the healthy rate (OFF bar unchanged) and
find the smallest repair with the full defender move set (greedy and robust). Answers "can it be fixed properly, not
just above threshold?"
Usage: python recovery.py --sex female --behaviour eat [--levels 40 60 80 100]
"""
import argparse, json, os, time
import numpy as np
from brain import Brain
from behaviours import Behaviour, DEFENCE_RULES
from agents import Attacker, Defender, Log, CHECK_SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--levels", type=int, nargs="+", default=[40, 60, 80, 100], help="required ON rate as percent of healthy")
args = ap.parse_args()


def _iv(d):
    out = {}
    for k, v in (d or {}).items():
        if k == "lesion": out[k] = [int(x) for x in v]
        elif k == "potentiate": out[k] = {tuple(int(t) for t in key.strip("()").split(",")): float(f) for key, f in v.items()}
        else: out[k] = {int(a): float(b) for a, b in v.items()}
    return out


brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); log = Log(quiet=True); t0 = time.time()
entry = json.load(open(f"atlas/{args.sex}/{args.behaviour}.json"))["entry"]
best = entry["goals"]["block"]["best"]; attack = {"intervention": _iv(best["intervention"]), "size": best["size"]}
healthy = float(np.mean([b.measure(True, seed=s)[0] for s in CHECK_SEEDS]))
attacked = float(np.mean([b.measure(True, seed=s, **attack["intervention"])[0] for s in CHECK_SEEDS]))
print(f"{args.sex} {b.name}: healthy {healthy:.0f} Hz; attack '{best['strategy']}' on {best['size']} neurons -> {attacked:.0f} Hz\n")
print(f"{'required':>9s} {'ON bar':>7s} | {'greedy repair':>14s} {'reached':>8s} | {'robust repair':>14s} {'reached':>8s} {'fragile':>8s}")
rows = []
for lvl in args.levels:
    b.on_thr = round(healthy * lvl / 100.0, 1)
    g = Defender(b, attack, log).greedy(name=f"rec{lvl}-greedy")
    r = Defender(b, attack, log).robust(name=f"rec{lvl}-robust")
    row = dict(level_pct=lvl, on_bar_hz=b.on_thr,
               greedy_size=g["size"] if g else None, greedy_hz=round(g["on_hz"]) if g else None,
               robust_size=r["size"] if r else None, robust_hz=round(r["on_hz"]) if r else None, robust_fragile=r.get("fragile") if r else None,
               robust_move=r["move"] if r else None)
    rows.append(row)
    print(f"{lvl:>8d}% {b.on_thr:>7.0f} | {str(row['greedy_size']):>14s} {str(row['greedy_hz']):>8s} | {str(row['robust_size']):>14s} {str(row['robust_hz']):>8s} {str(row['robust_fragile']):>8s}   {row['robust_move'] or ''}", flush=True)
os.makedirs(f"atlas/{args.sex}", exist_ok=True)
json.dump(dict(sex=args.sex, behaviour=b.name, healthy_hz=healthy, attacked_hz=attacked, attack=best, curve=rows,
               seconds=round(time.time() - t0)), open(f"atlas/{args.sex}/recovery_{b.name}.json", "w"), indent=1, default=str)
print(f"\nsaved atlas/{args.sex}/recovery_{b.name}.json  ({time.time()-t0:.0f}s)")
