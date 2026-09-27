"""Audit (27 Sep, exploratory): re-derive every headline number from the raw result files, plus reviewer-style
sensitivity checks. Reads files only; no simulation. Output: results/audit/audit.json"""
import glob, json, os
import numpy as np
from scipy.stats import binomtest, spearmanr
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); A = {}
L = lambda p: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, p)))]
fly = lambda n: n.split("_")[2]; grp = lambda n: int(n.split("_")[1])

# 1. screen counts
B = json.load(open(os.path.join(R, "screen", "stage_B.json")))
A["screen"] = dict(tested=len(B), included=sum(r["n_flies_pass"] >= 2 for r in B), all3=sum(r["n_flies_pass"] == 3 for r in B))
# 2. attack/repair per fly (primary)
S = L("instances/s_*_block.json"); A["per_fly"] = {}
for f in ("FlyWire", "BANC", "male"):
    rs = [r for r in S if fly(r["instance"]) == f]; at = [r for r in rs if r.get("attack")]; rp = [r for r in at if r.get("repair")]
    A["per_fly"][f] = dict(runs=len(rs), attack=len(at), grades={g: sum(r["attack_grade"] == g for r in at) for g in ("SOLID", "MARGINAL", "FAIL")},
                           median_attack=float(np.median([r["attack"]["size"] for r in at])), repair=len(rp),
                           repair_solid=sum(r.get("repair_grade") == "SOLID" for r in rp), median_repair=float(np.median([r["repair"]["size"] for r in rp])),
                           healthy_off_max=max(r["healthy_off"] for r in rs),
                           repair_cheats=sum(bool(r.get("cheats")) for r in rp))
# 3. OFF condition: is 'without stimulus' ever non-zero anywhere?
A["off_condition_all_zero"] = all(r["healthy_off"] == 0 for r in S)
# 4. H3 clustered by output group (sensitivity for non-independence across flies)
d = {}
for r in S:
    if r.get("attack") and r.get("repair"): d.setdefault(grp(r["instance"]), []).append(r["repair"]["size"] - r["attack"]["size"])
g = [np.sign(np.mean(v)) for v in d.values()]; up, dn = sum(x > 0 for x in g), sum(x < 0 for x in g)
A["H3_by_group"] = dict(groups=len(g), repair_larger=up, smaller=dn, tied=len(g) - up - dn, p=binomtest(up, up + dn, 0.5, alternative="greater").pvalue)
A["H3_per_fly_p"] = {f: binomtest(sum(x > 0 for k, v in d.items() for x in []) or 0, 1).pvalue if False else None for f in ()}
for f in ("FlyWire", "BANC", "male"):
    x = [r["repair"]["size"] - r["attack"]["size"] for r in S if fly(r["instance"]) == f and r.get("attack") and r.get("repair")]
    u, w = sum(v > 0 for v in x), sum(v < 0 for v in x); A.setdefault("H3_per_fly", {})[f] = dict(larger=u, smaller=w, p=binomtest(u, u + w, 0.5, alternative="greater").pvalue)
# 5. H1: is R1 predicting cost among FOUND attacks only, or just found vs not found?
H = json.load(open(os.path.join(R, "h1", "h1_screen.json")))["rows"]
A["H1_found_only"] = {f: dict(n=len(x), rho=float(spearmanr([r["R1"] for r in x], [r["cost"] for r in x])[0]), p=float(spearmanr([r["R1"] for r in x], [r["cost"] for r in x])[1]))
                      for f in ("FlyWire", "BANC", "male") for x in [[r for r in H if r["fly"] == f and r["cost"] < 1000]]}
from scipy.stats import mannwhitneyu
fr = [r["R1"] for r in H if r["cost"] < 1000]; nf = [r["R1"] for r in H if r["cost"] >= 1000]
A["H1_R1_found_vs_not"] = dict(median_found=float(np.median(fr)), median_not_found=float(np.median(nf)), p=float(mannwhitneyu(nf, fr, alternative="greater").pvalue))
# 6. H2 clustered by group
H2 = json.load(open(os.path.join(R, "h2", "h2_h5_screen_summary.json")))["primary"]["rows"]
gg = {}
for r in H2: gg.setdefault(grp(r["name"]), []).append(r["supports"])
A["H2_by_group"] = dict(groups=len(gg), majority_support=sum(np.mean(v) > 0.5 for v in gg.values()), tie=sum(np.mean(v) == 0.5 for v in gg.values()))
# 7. robustness A/C so far
RB = L("robust/s_*_block.json")
ok = [r for r in RB if r["attack_grade"] in ("SOLID", "MARGINAL")]; okr = [r for r in RB if r.get("repair_grade") in ("SOLID", "MARGINAL")]
A["robust_A"] = dict(done=len(RB), attack_agree=f"{sum(r['A']['attack_ok'] >= 12 for r in ok)}/{len(ok)}", repair_agree=f"{sum(r['A'].get('repair_ok', 0) >= 12 for r in okr)}/{len(okr)}",
                     healthy_fullbrain_fail=[r["instance"] for r in RB if r["A"]["healthy_ok"] < 12],
                     attack_mismatch=[r["instance"] for r in ok if r["A"]["attack_ok"] < 12], repair_mismatch=[r["instance"] for r in okr if r["A"].get("repair_ok", 0) < 12])
c = [r["C"]["share_break"] for r in RB if r["C"]["share_break"] is not None]
A["robust_C"] = dict(n=len(c), median=float(np.median(c)), max=max(c), none_valid=[r["instance"] for r in RB if r["C"]["share_break"] is None])
json.dump(A, open(os.path.join(R, "audit", "audit.json"), "w"), indent=1, default=float)
print(json.dumps(A, indent=1, default=float))
