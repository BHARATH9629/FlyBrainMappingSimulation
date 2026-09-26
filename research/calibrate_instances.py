"""Inclusion check (pre-registration, section 2): does the healthy model pass each new instance's two-condition test
on all 8 check seeds (1-8)? Thresholds are fixed in research/instances.py before this runs; they are never adjusted.

Usage: python research/calibrate_instances.py
"""
import json, os, time
import numpy as np
from guard import start, finish
out = start("calibration")
import instances
from brain import Brain
from behaviours import Behaviour

SEEDS = range(1, 9)
import sys
brain = Brain("female"); names = instances.register(brain=brain); report = {}
if len(sys.argv) > 1: names = sys.argv[1:]
for n in names:
    t0 = time.time(); b = Behaviour(brain, n)
    on = [b.measure(True, seed=s)[0] for s in SEEDS]; off = [b.measure(False, seed=s)[0] for s in SEEDS]
    if b.expected == "off": passes = [a <= b.off_thr and c <= b.off_thr for a, c in zip(on, off)]
    else: passes = [a >= b.on_thr and c <= b.off_thr for a, c in zip(on, off)]
    ok = all(passes)
    report[n] = dict(role=instances.role(n), stim_neurons=len(b.stim), readouts=len(b.readouts), expected=b.expected,
                     on_thr=b.on_thr, off_thr=b.off_thr, with_stim=on, without_stim=off, passes=int(sum(passes)), included=ok)
    print(f"{n:11s} ({instances.role(n):9s}) stim {len(b.stim):3d}  with {np.mean(on):6.1f} Hz (min {min(on):5.1f}, max {max(on):5.1f})"
          f"  without {np.mean(off):4.1f} Hz  -> {sum(passes)}/8 {'INCLUDED' if ok else 'EXCLUDED'}  ({time.time() - t0:.0f}s)", flush=True)
fn = os.path.join(out, "calibration.json"); old = json.load(open(fn)) if os.path.exists(fn) else {}
json.dump(old | report, open(fn, "w"), indent=1)
finish()
