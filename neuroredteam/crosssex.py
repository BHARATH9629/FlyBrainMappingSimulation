"""Cross-sex matching: are the neurons the attacker / defender uses the same CELL TYPES in the female and the male?

Reads atlas/<sex>/<behaviour>.json (best attack, all strategies, repair) and, if present, chokepoints_<behaviour>.json,
maps every neuron to a cell type (female: Schlegel cell_type; male: MaleCNS flywireType, which is the matched FlyWire
type), and tabulates conserved (used in both sexes) vs sex-specific types.
Output: atlas/crosssex_<behaviour>.csv  (the one file where both sexes appear, columns prefixed female_ / male_)
Usage: python crosssex.py --behaviour eat
"""
import argparse, csv, json, os
import pandas as pd
from brain import Brain

ap = argparse.ArgumentParser(); ap.add_argument("--behaviour", default="eat"); args = ap.parse_args()


def neurons(iv):
    return set(iv.get("lesion", [])) | {int(k) for k in iv.get("inject", {})} | {int(k) for k in iv.get("weaken", {})} \
         | {int(k) for k in iv.get("bias", {})} | {int(k.strip("()").split(",")[0]) for k in iv.get("potentiate", {})}


rows = []
per_sex = {}
for sex in ("female", "male"):
    path = f"atlas/{sex}/{args.behaviour}.json"
    if not os.path.exists(path): print(f"no {path}"); continue
    brain = Brain(sex); T = brain.types
    ftype = T["flywireType"].astype(str).values; ttype = T["type"].astype(str).values
    entry = json.load(open(path))["entry"]
    used = {}                                                    # index -> set of roles
    for goal, g in entry["goals"].items():
        for strat, atk in g["attacks"].items():
            if atk:
                for i in neurons(atk["intervention"]): used.setdefault(i, set()).add(f"{goal}:{strat.replace('attacker-', '')}")
        if g.get("repair"):
            for i in neurons(g["repair"]["intervention"]): used.setdefault(i, set()).add("repair")
    cp = f"atlas/{sex}/chokepoints_{args.behaviour}.json"
    freq = {}
    if os.path.exists(cp):
        c = json.load(open(cp))
        for S in c["attacks"]:
            for i in S: freq[i] = freq.get(i, 0) + 1 / max(c["n_minimal_attacks"], 1)
        for i in freq: used.setdefault(i, set()).add("chokepoint")
    per_sex[sex] = {}
    for i, roles in used.items():
        key = ftype[i] if ftype[i] not in ("nan", "") else ttype[i]
        per_sex[sex].setdefault(key, []).append(dict(index=i, id=int(brain.ids[i]), local_type=ttype[i], roles=sorted(roles),
                                                    chokepoint_freq=round(freq.get(i, 0.0), 2)))

types = sorted(set(per_sex.get("female", {})) | set(per_sex.get("male", {})))
out = f"atlas/crosssex_{args.behaviour}.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["matched_type", "status", "female_ids", "female_roles", "female_chokepoint_freq", "male_ids", "male_local_type", "male_roles", "male_chokepoint_freq"])
    n_cons = 0
    for t in types:
        fe, ma = per_sex.get("female", {}).get(t, []), per_sex.get("male", {}).get(t, [])
        status = "conserved" if fe and ma else ("female only" if fe else "male only")
        n_cons += bool(fe and ma)
        w.writerow([t, status, ";".join(str(x["id"]) for x in fe), ";".join(sorted({r for x in fe for r in x["roles"]})),
                    max([x["chokepoint_freq"] for x in fe], default=""), ";".join(str(x["id"]) for x in ma),
                    ";".join(sorted({x["local_type"] for x in ma})), ";".join(sorted({r for x in ma for r in x["roles"]})),
                    max([x["chokepoint_freq"] for x in ma], default="")])
print(f"{args.behaviour}: {len(types)} matched types used by attacker/defender in either sex, {n_cons} conserved (used in both)")
for t in types:
    fe, ma = per_sex.get("female", {}).get(t, []), per_sex.get("male", {}).get(t, [])
    tag = "BOTH  " if fe and ma else ("female" if fe else "male  ")
    print(f"  {tag} {t:12s} F:{','.join(sorted({r for x in fe for r in x['roles']})) or '-':40s} M:{','.join(sorted({r for x in ma for r in x['roles']})) or '-'}")
print(f"saved {out}")
