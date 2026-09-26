"""Inclusion check for the BANC behaviours (pre-registration, section 7): does the healthy BANC model pass each
behaviour's two-condition test on all 8 check seeds? Primary: weights x 1.815 (pre-registered); secondary: unscaled.

Usage: python research/calibrate_banc.py
"""
import json, os, time
import numpy as np
from guard import start, finish
out = start("banc")
import instances
from banc import BancBrain
from behaviours import Behaviour

SEEDS = range(1, 9); SCALE = json.load(open(os.path.join(out, "scale_factor.json")))["median_ratio"]
names = instances.register_banc(); report = {}
for label, scale in (("scaled", SCALE), ("unscaled", 1.0)):
    brain = BancBrain(scale=scale); report[label] = {"scale": scale}
    print(f"--- BANC, weights x {scale:.3f} ({label})", flush=True)
    for n in names:
        t0 = time.time(); b = Behaviour(brain, n)
        on = [b.measure(True, seed=s)[0] for s in SEEDS]; off = [b.measure(False, seed=s)[0] for s in SEEDS]
        if b.expected == "off": passes = [a <= b.off_thr and c <= b.off_thr for a, c in zip(on, off)]
        else: passes = [a >= b.on_thr and c <= b.off_thr for a, c in zip(on, off)]
        report[label][n] = dict(stim_neurons=len(b.stim), readouts=len(b.readouts), with_stim=on, without_stim=off,
                                passes=int(sum(passes)), included=all(passes))
        print(f"{n:11s} stim {len(b.stim):3d} readouts {len(b.readouts)}  with {np.mean(on):6.1f} Hz (min {min(on):5.1f})"
              f"  without {np.mean(off):4.1f}  -> {sum(passes)}/8 {'INCLUDED' if all(passes) else 'EXCLUDED'}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(report, open(os.path.join(out, "calibration_banc.json"), "w"), indent=1)
finish()
