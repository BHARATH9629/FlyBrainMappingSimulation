"""Check: the s_ instances resolve to exactly the neurons research/screen.py used, and the healthy rate on seed 1
matches the screen's stage B value for that seed."""
import json, os, sys
import numpy as np
from guard import start, finish
out = start("screen")
import instances
from behaviours import Behaviour
fly = sys.argv[1]
if fly == "BANC":
    from banc import BancBrain; brain = BancBrain(scale=json.load(open(os.path.join(os.path.dirname(out), "banc", "scale_factor.json")))["median_ratio"])
elif fly == "male":
    from malecns import MaleBrain; brain = MaleBrain(scale=json.load(open(os.path.join(os.path.dirname(out), "male", "scale_factor.json")))["median_ratio"])
else:
    from brain import Brain; brain = Brain("female")
names = instances.register_screen(brain, fly)
B = {(r["input"], r["output"]): r for r in json.load(open(os.path.join(out, "stage_B.json")))}
jobs = {n: (i, o) for n, f, i, o in instances.screen_jobs()}
bad = 0
for n in names[:3]:
    b = Behaviour(brain, n); on = float(np.mean([b.measure(True, seed=s)[0] for s in range(1, 9)]))
    ref = B[jobs[n]]["flies"][fly]["mean_on"]; same = abs(on - ref) < 1e-9; bad += not same
    print(f"{n}: stim {len(b.stim)} readouts {len(b.readouts)} mean rate seeds 1-8 {on:.4f} vs screen {ref:.4f} -> {'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
print(f"{fly}: {'ALL IDENTICAL' if not bad else str(bad) + ' DIFFERENT'}")
finish()
