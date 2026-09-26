"""Systematic pathway screen (pre-registration, section 7, 26 Sep 13:07).

Stage A (--stage A): for every input cell type (sensory or visual projection, same name in all three flies, 10-300
neurons on the left), drive its LEFT neurons at 150 Hz once (seed 1) in each fly's primary model and record the
descending neuron (by type + side) with the highest rate.
Stage B (--stage B): every (input type, output type + side) found in stage A is tested in all three flies on seeds 1-8
(two-condition test, >= 40 Hz with, <= 10 Hz without); included if it passes in at least two flies.

Names are matched across flies by the FlyWire name (FlyWire: its own; BANC: FlyWire-matched type; male: flywireType).
Usage: python research/screen.py --stage A      (then --stage B)
"""
import json, os, sys, time
import numpy as np
from guard import start, finish
out = start("screen")
from brain import Brain
from banc import BancBrain
from malecns import MaleBrain

STAGE = sys.argv[sys.argv.index("--stage") + 1]
SC = {"BANC": json.load(open(os.path.join(os.path.dirname(out), "banc", "scale_factor.json")))["median_ratio"],
      "male": json.load(open(os.path.join(os.path.dirname(out), "male", "scale_factor.json")))["median_ratio"]}
INPUT_SC = {"sensory", "visual_projection", "cb_sensory", "ol_sensory", "vnc_sensory"}
DN_SC = {"descending", "descending_neuron"}


def load(name):
    if name == "FlyWire": return Brain("female")
    if name == "BANC": return BancBrain(scale=SC["BANC"])
    return MaleBrain(scale=SC["male"])


def fw_names(brain, name):
    col = "flywireType" if name == "male" else "type"
    v = np.array([str(x) for x in brain.types[col]])
    if name == "male":                                   # fall back to the male's own name when there is no FlyWire one
        own = np.array([str(x) for x in brain.types["type"]]); v = np.where(np.isin(v, ["nan", "None", ""]), own, v)
    return v


FLIES = ("FlyWire", "BANC", "male")
brains = {f: load(f) for f in FLIES}
names = {f: fw_names(b, f) for f, b in brains.items()}
sides = {f: b.types["side"].astype(str).values for f, b in brains.items()}
supers = {f: np.array([str(x) for x in b.types["superclass"]]) for f, b in brains.items()}


def neurons(f, typ, side): return np.flatnonzero((names[f] == typ) & (sides[f] == side))


if STAGE == "A":
    per = {}
    for f in FLIES:
        m = np.isin(supers[f], list(INPUT_SC)) & (sides[f] == "L")
        t, c = np.unique(names[f][m], return_counts=True); per[f] = {a: int(b) for a, b in zip(t, c) if 10 <= b <= 300 and a not in ("nan", "None", "")}
    cands = sorted(set(per["FlyWire"]) & set(per["BANC"]) & set(per["male"]))
    print(f"input types in all three flies (10-300 left neurons): {len(cands)}", flush=True)
    res = {}
    for f in FLIES:
        b = brains[f]; dn = np.flatnonzero(np.isin(supers[f], list(DN_SC))); t0 = time.time()
        for k, typ in enumerate(cands):
            stim = neurons(f, typ, "L"); r = b.run(list(stim), record=[], seed=1)["active"]
            best = max(((hz, i) for i, hz in r.items() if i in set(dn.tolist())), default=(0.0, None))
            res.setdefault(typ, {})[f] = (dict(output=f"{names[f][best[1]]}|{sides[f][best[1]]}", hz=best[0]) if best[1] is not None else None)
            if k % 20 == 0: print(f"  {f}: {k}/{len(cands)}  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(out, "stage_A.json"), "w"), indent=1)
    n_any = sum(1 for v in res.values() if any(x and x["hz"] >= 40 for x in v.values()))
    print(f"stage A done: {len(res)} input types; {n_any} drive some descending neuron to >= 40 Hz in at least one fly")
else:
    A = json.load(open(os.path.join(out, "stage_A.json")))
    cands = sorted({(typ, v["output"]) for typ, d in A.items() for v in d.values() if v and v["hz"] >= 40})
    print(f"stage B: {len(cands)} candidate pathways x 3 flies x 8 seeds", flush=True)
    res = []
    for typ, outp in cands:
        otyp, oside = outp.split("|"); row = dict(input=typ, output=outp, flies={})
        for f in FLIES:
            stim, ro = neurons(f, typ, "L"), neurons(f, otyp, oside)
            if len(stim) == 0 or len(ro) == 0: row["flies"][f] = None; continue
            on = [float(np.mean([brains[f].run(list(stim), record=list(ro), seed=s)["rates"][i] for i in ro])) for s in range(1, 9)]
            off = [float(np.mean([brains[f].run([], record=list(ro), seed=s)["rates"][i] for i in ro])) for s in range(1, 9)]
            row["flies"][f] = dict(passes=int(sum(a >= 40 and c <= 10 for a, c in zip(on, off))), mean_on=float(np.mean(on)))
        row["n_flies_pass"] = sum(1 for v in row["flies"].values() if v and v["passes"] == 8); res.append(row)
        print(f"  {typ:14s} -> {outp:16s} " + "  ".join(f"{f}:{(v['passes'] if v else '-')}/8" for f, v in row["flies"].items())
              + f"  {'INCLUDED' if row['n_flies_pass'] >= 2 else ''}", flush=True)
    json.dump(res, open(os.path.join(out, "stage_B.json"), "w"), indent=1)
    print(f"stage B done: {sum(r['n_flies_pass'] >= 2 for r in res)} pathways pass in at least two flies")
finish()
