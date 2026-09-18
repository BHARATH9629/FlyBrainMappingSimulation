"""Batch runner: repeat the red/blue pipeline over behaviours and independent seed-sets, write the atlas as CSV + table.
Usage: python batch.py [--behaviours eat] [--reps 3]
Each rep uses a different triple of check seeds, so the atlas reports mean +/- sd of attack size, repair size, recovery."""
import argparse, json, os, csv, time
import numpy as np
from brain import Brain
from behaviours import Behaviour, BEHAVIOURS
from agents import Attacker, Defender, Log, redundancy

ap = argparse.ArgumentParser()
ap.add_argument("--behaviours", nargs="+", default=list(BEHAVIOURS)); ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()

brain = Brain(); rows = []; t0 = time.time()
for name in args.behaviours:
    b = Behaviour(brain, name)
    for rep in range(args.reps):
        seeds = tuple(10 * rep + s for s in (1, 2, 3))
        log = Log(quiet=True)
        healthy = float(np.mean([b.measure(True, seed=s)[0] for s in seeds]))
        found = [a for a in (Attacker(b, log, "block", seeds=seeds).greedy(),
                             Attacker(b, log, "block", seeds=seeds).inject_inhibitors(),
                             Attacker(b, log, "block", seeds=seeds).weaken_only()) if a]
        best = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
        force = Attacker(b, log, "force", seeds=seeds).greedy()
        disjoint = [s["size"] for s in redundancy(b, log, "block")]
        repair = Defender(b, best, log, seeds=seeds).greedy() if best else None
        row = dict(behaviour=name, rep=rep, seeds=str(seeds), healthy_hz=round(healthy, 1),
                   block_size=best["size"] if best else None, block_strategy=best["strategy"] if best else None,
                   block_sims=best["simulations"] if best else None,
                   force_size=force["size"] if force else None, force_sims=force["simulations"] if force else None,
                   n_disjoint_attacks=len(disjoint), disjoint_sizes=str(disjoint),
                   repair_size=repair["size"] if repair else None, repair_move=repair["move"] if repair else None,
                   recovery_pct=round(100 * repair["on_hz"] / healthy) if repair else None,
                   repair_sims=repair["simulations"] if repair else None, seconds=round(time.time() - t0))
        rows.append(row); print(row)
        json.dump(log.rows, open(f"atlas/log_{name}_rep{rep}.json", "w"), indent=1, default=str)

os.makedirs("atlas", exist_ok=True)
with open("atlas/atlas.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\n== FRAGILITY ATLAS (mean +/- sd over reps) ==")
print(f"{'behaviour':10s} {'healthy Hz':>10s} {'block size':>11s} {'force size':>11s} {'disjoint':>9s} {'repair size':>12s} {'recovery %':>11s}")
for name in args.behaviours:
    R = [r for r in rows if r["behaviour"] == name]
    def ms(k):
        v = [r[k] for r in R if r[k] is not None]
        return f"{np.mean(v):.1f} +/- {np.std(v):.1f}" if v else "none"
    print(f"{name:10s} {ms('healthy_hz'):>10s} {ms('block_size'):>11s} {ms('force_size'):>11s} {ms('n_disjoint_attacks'):>9s} {ms('repair_size'):>12s} {ms('recovery_pct'):>11s}")
print("saved atlas/atlas.csv")
