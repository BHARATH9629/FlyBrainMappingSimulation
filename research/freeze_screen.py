"""Freeze the screen result (pre-registration, 26 Sep): the included pathways, grouped by output neuron (type + side),
and ONE pathway per group chosen at random with a fixed seed (0) for the attack/repair runs. No simulation here."""
import json, os
import numpy as np
from guard import start, finish
out = start("screen")
B = json.load(open(os.path.join(out, "stage_B.json")))
inc = sorted([r for r in B if r["n_flies_pass"] >= 2], key=lambda r: (r["output"], r["input"]))
groups = {}
for r in inc: groups.setdefault(r["output"], []).append(r)
rng = np.random.default_rng(0); attack_set = []
for o in sorted(groups):
    g = groups[o]; r = g[int(rng.integers(len(g)))]
    attack_set.append(dict(input=r["input"], output=o, group_size=len(g),
                           flies=[f for f, v in r["flies"].items() if v and v["passes"] == 8]))
json.dump(dict(included=[dict(input=r["input"], output=r["output"], n_flies_pass=r["n_flies_pass"]) for r in inc],
               groups={o: len(g) for o, g in groups.items()}, attack_set=attack_set, seed=0),
          open(os.path.join(out, "frozen.json"), "w"), indent=1)
print(f"{len(inc)} included pathways, {len(groups)} output groups; attack set:")
for a in attack_set: print(f"  {a['input']:16s} -> {a['output']:14s} (group of {a['group_size']:2d})  flies {a['flies']}")
print(f"fly-runs: {sum(len(a['flies']) for a in attack_set)}")
finish()
