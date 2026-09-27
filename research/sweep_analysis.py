"""Scaling sweep analysis (pre-registration log, 27 Sep, item a): break rate (share of included pathways with an
attack found) and median attack size per fly and scale; Fisher's exact test of BANC x1.0 vs FlyWire."""
import glob, json, os
import numpy as np
from scipy.stats import fisher_exact
from guard import start, finish
out = start("sweep")
R = os.path.dirname(out)
sets = {("FlyWire", 1.0): "instances/s_*_FlyWire_block.json", ("BANC", 1.815): "instances/s_*_BANC_block.json",
        ("male", 0.556): "instances/s_*_male_block.json", ("BANC", 1.0): "sweep_BANC_x1.0/*.json",
        ("BANC", 1.4): "sweep_BANC_x1.4/*.json", ("male", 1.0): "sweep_male_x1.0/*.json"}
rep = {}
for (fly, sc), pat in sorted(sets.items()):
    rs = [json.load(open(f)) for f in glob.glob(os.path.join(R, pat))]
    inc = [r for r in rs if r.get("included", True)]; att = [r for r in inc if r.get("attack")]
    rep[f"{fly} x{sc}"] = d = dict(tested=len(rs), included=len(inc), attack_found=len(att), break_rate=round(len(att) / len(inc), 3),
                                   solid=sum(r["attack_grade"] == "SOLID" for r in att),
                                   median_size=float(np.median([r["attack"]["size"] for r in att])) if att else None)
    print(f"{fly:8s} x{sc:<6} included {len(inc):2d}/{len(rs):2d}  attack found {len(att):2d} ({100 * d['break_rate']:.0f}%)  SOLID {d['solid']:2d}  median size {d['median_size']}")
a, b = rep["BANC x1.0"], rep["FlyWire x1.0"]
odds, p = fisher_exact([[a["attack_found"], a["included"] - a["attack_found"]], [b["attack_found"], b["included"] - b["attack_found"]]])
rep["fisher_BANC1.0_vs_FlyWire"] = dict(odds=odds, p=p); print(f"Fisher BANC x1.0 vs FlyWire: odds {odds:.2f}, p = {p:.3f}")
a = rep["BANC x1.815"]
odds2, p2 = fisher_exact([[a["attack_found"], a["included"] - a["attack_found"]], [b["attack_found"], b["included"] - b["attack_found"]]])
rep["fisher_BANC1.815_vs_FlyWire"] = dict(odds=odds2, p=p2); print(f"(for reference) BANC x1.815 vs FlyWire: odds {odds2:.2f}, p = {p2:.4f}")
json.dump(rep, open(os.path.join(out, "sweep_summary.json"), "w"), indent=1)
finish()
