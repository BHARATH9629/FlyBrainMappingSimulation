"""Calibrate a behaviour: what does the healthy brain do with this stimulus?

Prints, for left / right / both stimulus sides on seeds 1-3: the readout rate with and without the stimulus, the
sub-network size, and the most active descending / motor neurons (by type) with the stimulus. Use it to choose a
readout for a new behaviour and to set its ON / OFF thresholds; then record the numbers in behaviours.py.
Usage: python calibrate.py --sex female --behaviour escape [--top 15] [--full]
"""
import argparse, numpy as np
from collections import defaultdict
from brain import Brain
from behaviours import Behaviour, BEHAVIOURS

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", required=True)
ap.add_argument("--top", type=int, default=15); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--full", action="store_true", help="also run the whole reachable brain (depth 99, min_syn 1)")
args = ap.parse_args()

brain = Brain(args.sex); cfg = BEHAVIOURS[args.sex][args.behaviour]
T = brain.types
print(f"{args.sex} fly, behaviour '{args.behaviour}': {cfg.get('description', '')}\n  source: {cfg.get('source', '(none given)')}")

for side in ("L", "R", None):
    try: b = Behaviour(brain, args.behaviour, side=side)
    except Exception as e: print(f"\n[{side or 'both'}] cannot resolve: {e}"); continue
    ons, offs, n_sub, acts = [], [], None, defaultdict(list)
    for s in args.seeds:
        on, r = b.measure(True, seed=s); off, _ = b.measure(False, seed=s); ons.append(on); offs.append(off); n_sub = r["n_sub"]
        for i, hz in r["active"].items(): acts[i].append(hz)
    label = {"L": "left", "R": "right", None: "both sides"}[side]
    print(f"\n[{label}] stimulus {len(b.stim)} neurons, readout {len(b.readouts)} neuron(s) {[brain.type_of(i) for i in b.readouts][:4]}"
          f"  sub-network {n_sub:,}")
    print(f"  readout: {np.mean(ons):6.1f} +/- {np.std(ons):4.1f} Hz with stimulus, {np.mean(offs):6.1f} Hz without"
          f"   (thresholds in spec: on >= {b.on_thr}, off <= {b.off_thr}; expected {b.expected})")
    # most active descending / motor neurons with the stimulus (candidates for a readout)
    if T is not None:
        sup = np.asarray(T["superclass"].fillna("").astype(str)); sides = np.asarray(T["side"].fillna("?").astype(str))
        cand = [(i, np.mean(v)) for i, v in acts.items() if ("descending" in sup[i] or "motor" in sup[i]) and i not in set(b.stim)]
        cand.sort(key=lambda kv: -kv[1])
        print(f"  most active descending / motor neurons with the stimulus (mean Hz over seeds):")
        for i, hz in cand[:args.top]:
            print(f"    {i:>7d}  {brain.type_of(i) or '?':16s} {sup[i]:20s} side {sides[i]}  {hz:6.1f} Hz")
    if args.full:
        onf = [b.measure(True, seed=s, depth=99, min_syn=1)[0] for s in args.seeds[:1]]
        print(f"  FULL BRAIN (seed {args.seeds[0]}): readout {onf[0]:.1f} Hz with stimulus")
