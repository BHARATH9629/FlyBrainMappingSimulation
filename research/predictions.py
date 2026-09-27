"""Biological predictions (27 Sep): cell types that the red team used to break the SAME pathway in two or more flies
(primary scales, plus BANC x1.0). For each: the pathway (input -> output), the move (silence = lesion/weaken, or
activate = inject), transmitter sign, and in which flies. These are testable with genetic silencing/activation.
Output: results/predictions/predictions.json + .md"""
import glob, json, os, collections
import numpy as np
from guard import start, finish
out = start("predictions")
import instances
R = os.path.dirname(out); BAD = {"nan", "None", ""}
fz = json.load(open(os.path.join(R, "screen", "frozen.json")))["attack_set"]
SRC = {"FlyWire": ["instances"], "BANC": ["instances", "sweep_BANC_x1.0"], "male": ["instances"]}
use = collections.defaultdict(lambda: collections.defaultdict(set))    # (group, type) -> fly -> {moves}
nt = {}
for f in ("FlyWire", "BANC", "male"):
    b = instances.screen_brain(f); names = instances.screen_names(b, f); W = b.W.tocsr()
    for d in SRC[f]:
        for fn in glob.glob(os.path.join(R, d, f"s_*_{f}_block.json")):
            r = json.load(open(fn))
            if not (r.get("included", True) and r.get("attack")): continue
            k = int(r["instance"].split("_")[1]); iv = r["attack"]["intervention"]
            for mv, ids in (("silence", iv.get("lesion", [])), ("silence", list(iv.get("weaken", {}))), ("activate", list(iv.get("inject", {})))):
                for i in ids:
                    i = int(i); t = names[i]
                    if t in BAD: continue
                    use[(k, t)][f].add(mv + ("" if d == "instances" else " (BANC x1.0)"))
                    row = W[i]; s = np.sign(row.data.sum()) if row.nnz else 0
                    nt.setdefault(t, {})[f] = "inhibitory" if s < 0 else "excitatory" if s > 0 else "?"
    del b
P = []
for (k, t), flies in use.items():
    if len(flies) < 2: continue
    a = fz[k]; moves = sorted({m.split(" (")[0] for v in flies.values() for m in v})
    P.append(dict(group=k, input=a["input"], output=a["output"], cell_type=t, flies=sorted(flies), n_flies=len(flies),
                  move=moves, transmitter=nt.get(t, {}), detail={f: sorted(v) for f, v in flies.items()}))
P.sort(key=lambda p: (-p["n_flies"], p["group"], p["cell_type"]))
json.dump(P, open(os.path.join(out, "predictions.json"), "w"), indent=1)
lines = ["| Pathway (left input -> output) | Weak-spot cell type | Move that breaks it | Transmitter (sign) | Flies |", "|---|---|---|---|---|"]
for p in P:
    tr = "/".join(sorted(set(p["transmitter"].values())))
    lines.append(f"| {p['input']} -> {p['output']} | **{p['cell_type']}** | {' / '.join(p['move'])} | {tr} | {', '.join(p['flies'])} |")
open(os.path.join(out, "predictions.md"), "w").write("\n".join(lines) + "\n")
print(f"{len(P)} predictions ({sum(p['n_flies'] == 3 for p in P)} in all three flies)"); print("\n".join(lines))
finish()
