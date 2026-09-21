"""Threshold sensitivity: do the atlas numbers depend on our choice of ON / OFF cut-offs (default 40 / 10 Hz)?

Re-runs the full attacker set (block + force), redundancy and the defender for several threshold pairs, on the
CHECK_SEEDS, and writes a table. Because the brain results are cached in memory, later threshold pairs are cheap.
Usage: python sensitivity.py --sex female [--behaviour eat]
"""
import argparse, csv, json, os, time
import numpy as np
from brain import Brain
from behaviours import Behaviour
from agents import Attacker, Defender, Log, redundancy

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--thresholds", nargs="+", default=["30/15", "40/10", "50/10", "60/5"], help="on/off Hz pairs")
args = ap.parse_args()

brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); log = Log(quiet=True)
if b.expected == "off": raise SystemExit("sensitivity.py is for block/force behaviours (expected on)")
healthy = float(np.mean([b.measure(True, seed=s)[0] for s in (1, 2, 3)]))
rows = []
print(f"{args.sex} fly, {b.name}: healthy readout {healthy:.1f} Hz\n")
print(f"{'on/off Hz':>10s} | {'block size':>10s} {'strategy':>18s} | {'force':>5s} | {'disjoint':>8s} | {'repair':>6s} {'move':>18s} {'recovery':>8s}")
for th in args.thresholds:
    on_thr, off_thr = (float(x) for x in th.split("/"))
    b.on_thr, b.off_thr = on_thr, off_thr; t0 = time.time()
    found = [a for a in (Attacker(b, log, "block").greedy(), Attacker(b, log, "block").inject_inhibitors(),
                         Attacker(b, log, "block").weaken_only()) if a]
    best = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
    force = Attacker(b, log, "force").greedy()
    disjoint = [s["size"] for s in redundancy(b, log, "block")]
    repair = Defender(b, best, log).greedy() if best else None
    row = dict(sex=args.sex, behaviour=b.name, on_thr=on_thr, off_thr=off_thr, healthy_hz=round(healthy, 1),
               block_size=best["size"] if best else None, block_strategy=best["strategy"] if best else None,
               force_size=force["size"] if force else None, n_disjoint=len(disjoint), disjoint_sizes=disjoint,
               repair_size=repair["size"] if repair else None, repair_move=repair["move"] if repair else None,
               recovery_pct=round(100 * repair["on_hz"] / healthy) if repair else None, seconds=round(time.time() - t0))
    rows.append(row)
    print(f"{th:>10s} | {str(row['block_size']):>10s} {str(row['block_strategy']):>18s} | {str(row['force_size']):>5s} | "
          f"{str(disjoint):>8s} | {str(row['repair_size']):>6s} {str(row['repair_move']):>18s} {str(row['recovery_pct']):>7s}%   ({row['seconds']}s)", flush=True)

os.makedirs(f"atlas/{args.sex}", exist_ok=True)
with open(f"atlas/{args.sex}/sensitivity_{b.name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\nsaved atlas/{args.sex}/sensitivity_{b.name}.csv")
