"""Robustness controls A (whole-brain fidelity) and C (random attacks of the same size) for ONE screen fly-run
(pre-registration log, 27 Sep). Uses the unchanged simulator. Output: results/robust/<name>.json
Usage: python research/robust_ac.py s_02_FlyWire block"""
import json, os, sys, time
import numpy as np
from guard import start, finish
out = start("robust")
import instances
from behaviours import Behaviour
from agents import merge_iv
NAME = sys.argv[1]; FLY = NAME.split("_")[2]; FRESH = tuple(range(11, 26)); FULL = dict(depth=99, min_syn=1)
R = os.path.dirname(out)
base = json.load(open(os.path.join(R, "instances", f"{NAME}_block.json")))


def ivin(d):
    o = {}
    for k, v in (d or {}).items():
        if k == "lesion": o[k] = [int(x) for x in v]
        elif k == "potentiate": o[k] = {tuple(int(t) for t in key.split(",")): float(f) for key, f in v.items()}
        else: o[k] = {int(a): float(b) for a, b in v.items()}
    return o


brain = instances.screen_brain(FLY); instances.register_screen(brain, FLY); b = Behaviour(brain, NAME)
t0 = time.time(); row = dict(instance=NAME, attack_grade=base.get("attack_grade"), repair_grade=base.get("repair_grade"))
atk = ivin(base["attack"]["intervention"])
rate = lambda stim, iv, **kw: [b.measure(stim, seed=s, **iv, **kw)[0] for s in FRESH]
# ---- A: whole brain
h = rate(True, {}, **FULL); a = rate(True, atk, **FULL)
row["A"] = dict(healthy_ok=int(sum(x >= b.on_thr for x in h)), attack_ok=int(sum(x <= b.off_thr for x in a)), healthy_hz=h, attack_hz=a)
if base.get("repair"):
    full = merge_iv(atk, ivin(base["repair"]["intervention"])); rr = rate(True, full, **FULL)
    row["A"].update(repair_ok=int(sum(x >= b.on_thr for x in rr)), repair_hz=rr)
# ---- C: random attacks of the same size and move type(s) from the readout's direct partners
W = brain.W.tocsc(); rng = np.random.default_rng(0); ro = set(b.readouts)
col = {}
for r in b.readouts:
    c = W[:, r]
    for i, w in zip(c.indices, c.data): col[int(i)] = col.get(int(i), 0) + w
direct = [i for i, w in col.items() if abs(w) >= 5 * getattr(brain, "scale", 1.0) and i not in ro]
inhib = [i for i in direct if col[i] < 0]
sizes = {k: (len(v) if isinstance(v, (list, dict)) else 0) for k, v in atk.items()}
C = []
for d in range(20):
    iv = {}; used = set()
    for k, n in sizes.items():
        pool = [i for i in (inhib if k == "inject" else direct) if i not in used]
        if len(pool) < n: iv = None; break
        pick = [int(x) for x in rng.choice(pool, n, replace=False)]; used |= set(pick)
        iv[k] = pick if k == "lesion" else {i: list(atk[k].values())[j] for j, i in enumerate(pick)}
    if iv is None: C.append(None); continue
    rr = rate(True, iv); C.append(int(sum(x <= b.off_thr for x in rr)))
valid = [c for c in C if c is not None]
row["C"] = dict(n_direct=len(direct), n_inhib=len(inhib), draws=C, share_break=(sum(c >= 12 for c in valid) / len(valid)) if valid else None)
row["seconds"] = round(time.time() - t0)
json.dump(row, open(os.path.join(out, f"{NAME}_block.json"), "w"), indent=1)
A = row["A"]
print(f"{NAME} block: A healthy {A['healthy_ok']}/15 attack {A['attack_ok']}/15" + (f" repair {A['repair_ok']}/15" if "repair_ok" in A else "")
      + f" | C random break share {row['C']['share_break']} | {row['seconds']}s", flush=True)
finish()
