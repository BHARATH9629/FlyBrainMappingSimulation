"""H3 on the screen data (pre-registration log, 27 Sep): is repair costlier than attack? Sign test on repair size minus
attack size over fly-runs with both; ties dropped. Primary: pre-registered scales; secondary: BANC x1.0."""
import glob, json, os
from scipy.stats import binomtest
from guard import start, finish
out = start("h3")
R = os.path.dirname(out)
SETS = {"primary": ["instances/s_*_FlyWire_block.json", "instances/s_*_BANC_block.json", "instances/s_*_male_block.json"],
        "secondary (BANC x1.0)": ["instances/s_*_FlyWire_block.json", "sweep_BANC_x1.0/*.json", "instances/s_*_male_block.json"]}
rep = {}
for label, pats in SETS.items():
    rows = []
    for p in pats:
        for f in sorted(glob.glob(os.path.join(R, p))):
            r = json.load(open(f))
            if r.get("attack") and r.get("repair"):
                rows.append(dict(name=r["instance"], attack=r["attack"]["size"], repair=r["repair"]["size"]))
    d = [x["repair"] - x["attack"] for x in rows]; up, down, tie = sum(v > 0 for v in d), sum(v < 0 for v in d), sum(v == 0 for v in d)
    p = binomtest(up, up + down, 0.5, alternative="greater").pvalue
    share = up / (up + down); ok = share >= 0.75 and p < 0.05
    per = {}
    for fly in ("FlyWire", "BANC", "male"):
        dd = [x["repair"] - x["attack"] for x in rows if x["name"].endswith(fly)]
        per[fly] = dict(n=len(dd), repair_larger=sum(v > 0 for v in dd), smaller=sum(v < 0 for v in dd), ties=sum(v == 0 for v in dd))
    rep[label] = dict(n=len(rows), repair_larger=up, repair_smaller=down, ties=tie, share=round(share, 3), p_one_sided=p,
                      confirmed=bool(ok), per_fly=per, rows=rows)
    print(f"[{label}] runs {len(rows)}: repair larger {up}, smaller {down}, tied {tie} -> {100 * share:.0f}% larger, p = {p:.2g}  H3 {'CONFIRMED' if ok else 'NOT confirmed'}")
    print("   per fly:", {k: (v["repair_larger"], v["smaller"], v["ties"]) for k, v in per.items()})
json.dump(rep, open(os.path.join(out, "h3_screen.json"), "w"), indent=1)
finish()
