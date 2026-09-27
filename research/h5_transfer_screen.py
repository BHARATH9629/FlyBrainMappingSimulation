"""H5 transfer on the screen data (pre-registration log, 27 Sep). An attack found in one fly is mapped by cell type
(FlyWire name, as in the screen; same side) onto the SAME pathway in another fly: every neuron of each attacked type on
that side receives the same move and strength. Tested on the 15 unseen seeds; "breaks" = goal met on >= 12/15.

  python h5_transfer_screen.py --map                 build results/h5/transfer_map.json (loads all three brains, no simulation)
  python h5_transfer_screen.py DEST [--scale S]      run every transfer INTO fly DEST (FlyWire / BANC / male)
Sets: primary = pre-registered scales; secondary = BANC x1.0 (as source and destination).
"""
import glob, json, os, sys
import numpy as np
from guard import start, finish
out = start("h5")
import instances
from behaviours import Behaviour
R = os.path.dirname(out); FRESH = tuple(range(11, 26)); FLIES = ("FlyWire", "BANC", "male"); BAD = {"nan", "None", ""}
SRC = {"primary": {"FlyWire": "instances", "BANC": "instances", "male": "instances"},
       "secondary": {"FlyWire": "instances", "BANC": "sweep_BANC_x1.0", "male": "instances"}}
MAP = os.path.join(out, "transfer_map.json")

if "--map" in sys.argv:
    M = {}
    for f in FLIES:
        b = instances.screen_brain(f); names = instances.screen_names(b, f); sides = b.types["side"].astype(str).values
        for label, src in SRC.items():
            for fn in sorted(glob.glob(os.path.join(R, src[f], f"s_*_{f}_block.json"))):
                r = json.load(open(fn))
                if not (r.get("included", True) and r.get("attack")): continue
                iv = r["attack"]["intervention"]; moves = []
                for i in iv.get("lesion", []): moves.append(("lesion", names[int(i)], sides[int(i)], None))
                for k in ("inject", "weaken", "bias"):
                    for i, v in (iv.get(k) or {}).items(): moves.append((k, names[int(i)], sides[int(i)], v))
                M.setdefault(label, {})[r["instance"]] = moves
        del b
    json.dump(M, open(MAP, "w"), indent=1); print({k: len(v) for k, v in M.items()}); finish(); sys.exit(0)

DEST = sys.argv[1]; SC = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else None
label = "secondary" if (DEST == "BANC" and SC == 1.0) else "primary"
M = json.load(open(MAP))
brain = instances.screen_brain(DEST, SC); instances.register_screen(brain, DEST)
names = instances.screen_names(brain, DEST); sides = brain.types["side"].astype(str).values
dest_included = {os.path.basename(p).split("_block")[0] for p in glob.glob(os.path.join(R, SRC[label][DEST], f"s_*_{DEST}_block.json"))
                 if json.load(open(p)).get("included", True)}
labels = [label] + (["secondary"] if label == "primary" and DEST != "BANC" else [])   # FlyWire/male receive from BANC x1.0 too
res = {}
for lab in labels:
    for src_name, moves in sorted(M[lab].items()):
        k, sfly = src_name.split("_")[1], src_name.split("_")[2]
        tgt = f"s_{k}_{DEST}"
        if sfly == DEST or tgt not in dest_included: continue
        if lab == "secondary" and sfly != "BANC" and label == "primary": continue       # only BANC x1.0 sources are new
        new = {}; dropped = set()
        for mv, ty, side, v in moves:
            idx = np.flatnonzero((names == ty) & (sides == side)) if ty not in BAD else []
            if len(idx) == 0: dropped.add(ty); continue
            for j in idx:
                if mv == "lesion": new.setdefault("lesion", []).append(int(j))
                else: new.setdefault(mv, {})[int(j)] = v
        b = Behaviour(brain, tgt)
        ro = set(b.readouts); new = {m: ([i for i in v if i not in ro] if isinstance(v, list) else {i: x for i, x in v.items() if i not in ro}) for m, v in new.items()}
        new = {m: v for m, v in new.items() if v}
        if not new:
            row = dict(n_neurons=0, dropped_types=sorted(dropped), goal_met=0, breaks=False, note="nothing to transfer")
        else:
            r = [b.measure(True, seed=s, **new)[0] for s in FRESH]; ok = [x <= b.off_thr for x in r]
            row = dict(n_neurons=sum(len(v) for v in new.values()), dropped_types=sorted(dropped), goal_met=int(sum(ok)),
                       breaks=bool(sum(ok) >= 12), mean_hz=float(np.mean(r)))
        res[f"{lab}|{src_name}->{tgt}"] = row
        print(f"[{lab}] {src_name} -> {tgt}: {row['goal_met']}/15 {'BREAKS' if row['breaks'] else 'no'} ({row['n_neurons']} neurons, dropped {row['dropped_types']})", flush=True)
json.dump(res, open(os.path.join(out, f"transfer_into_{DEST}{'_x%s' % SC if SC else ''}.json"), "w"), indent=1)
finish()
