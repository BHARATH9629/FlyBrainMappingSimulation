"""Summaries (pre-registered 27 Sep): H2 on the screen data (forbidden direct inputs: cost >= 2x baseline or not found,
in >= 75 % of runs with a baseline attack) and H5 transfer (breaks = goal met on >= 12/15; confirmed if >= 75 %).
Primary = pre-registered scales; secondary = BANC x1.0."""
import glob, json, os
from scipy.stats import binomtest
from guard import start, finish
out = start("h2")
R = os.path.dirname(out); rep = {}
for label, banc_base, banc_sfx in (("primary", "instances", ""), ("secondary (BANC x1.0)", "sweep_BANC_x1.0", "_x1.0")):
    rows = []
    for f in ("FlyWire", "BANC", "male"):
        base_dir = banc_base if f == "BANC" else "instances"; sfx = banc_sfx if f == "BANC" else ""
        for fn in sorted(glob.glob(os.path.join(R, base_dir, f"s_*_{f}_block.json"))):
            b = json.load(open(fn))
            if not (b.get("included", True) and b.get("attack")): continue
            h = os.path.join(out, os.path.basename(fn).replace(".json", f"{sfx}.json"))
            if not os.path.exists(h): continue
            h = json.load(open(h)); base = b["attack"]["size"]
            ok = (not h.get("found")) or (h["best_size"] >= 2 * base)
            rows.append(dict(name=b["instance"], baseline=base, forbidden=(h["best_size"] if h.get("found") else None), supports=bool(ok)))
    n, k = len(rows), sum(r["supports"] for r in rows)
    per = {f: (sum(r["supports"] for r in rows if r["name"].endswith(f)), sum(r["name"].endswith(f) for r in rows)) for f in ("FlyWire", "BANC", "male")}
    rep[label] = dict(n=n, supports=k, share=round(k / n, 3), confirmed=bool(k / n >= 0.75), per_fly=per,
                      not_found=sum(r["forbidden"] is None for r in rows), rows=rows)
    print(f"[H2 {label}] {k}/{n} ({100 * k / n:.0f}%) cost >= 2x or not found; not found {rep[label]['not_found']}; per fly {per} -> {'CONFIRMED' if k / n >= 0.75 else 'NOT confirmed'}")
T = {}
for fn in glob.glob(os.path.join(R, "h5", "transfer_into_*.json")): T.update({(os.path.basename(fn), k): v for k, v in json.load(open(fn)).items()})
for label, files, lab in (("primary", ("transfer_into_FlyWire.json", "transfer_into_male.json", "transfer_into_BANC.json"), "primary"),
                          ("secondary (BANC x1.0)", ("transfer_into_FlyWire.json", "transfer_into_male.json", "transfer_into_BANC_x1.0.json"), None)):
    rows = []
    for (fn, key), v in T.items():
        if fn not in files: continue
        lb, pair = key.split("|"); src = pair.split("->")[0]; srcfly = src.split("_")[2]
        if label == "primary" and lb != "primary": continue
        if label != "primary":
            # secondary: BANC sources from BANC x1.0, everything else primary; BANC destination at x1.0
            if srcfly == "BANC" and lb != "secondary": continue
            if srcfly != "BANC" and lb != ("secondary" if fn == "transfer_into_BANC_x1.0.json" else "primary"): continue
        rows.append(dict(pair=pair, into=fn, **v))
    n, k = len(rows), sum(r["breaks"] for r in rows); dropped = sum(bool(r["dropped_types"]) for r in rows)
    rep[f"H5 transfer {label}"] = dict(n=n, breaks=k, share=round(k / n, 3), confirmed=bool(k / n >= 0.75), with_dropped_types=dropped,
                                       breaks_without_dropped=f"{sum(r['breaks'] for r in rows if not r['dropped_types'])}/{sum(not r['dropped_types'] for r in rows)}")
    print(f"[H5 transfer {label}] {k}/{n} ({100 * k / n:.0f}%) break; {dropped} had a type that could not be mapped; "
          f"without those: {rep[f'H5 transfer {label}']['breaks_without_dropped']} -> {'CONFIRMED' if k / n >= 0.75 else 'NOT confirmed'}")
json.dump(rep, open(os.path.join(out, "h2_h5_screen_summary.json"), "w"), indent=1)
finish()
