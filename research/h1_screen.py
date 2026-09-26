"""H1 on the screen attack set (pre-registration log, 27 Sep, item b). Wiring measures R1/R2 from h1_wiring.py
(results/h1/wiring_screen_<fly>.json); attack results from results/instances/s_*_block.json.

Primary (as pre-registered 27 Sep): attack cost = size of the cheapest attack found (any grade); lesion-only cost =
size of the lesion (greedy) attack; runs with none found are ranked above every found cost (ties). Spearman per fly and
pooled; pooled one-sided p from 10,000 permutations shuffling cost within each fly. Confirmed if both pooled rho >= 0.6
with p < 0.05. Also reported (section 4 wording): the same with only SOLID-verified attacks counted as found.
"""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr
from guard import start, finish
out = start("h1")
FLIES = ("FlyWire", "BANC", "male"); NONE = 10 ** 3
rows = []
for f in FLIES:
    W = json.load(open(os.path.join(out, f"wiring_screen_{f}.json")))
    for fn in sorted(glob.glob(os.path.join(os.path.dirname(out), "instances", f"s_*_{f}_block.json"))):
        r = json.load(open(fn)); n = r["instance"]; a = r.get("attack")
        found = r.get("attacks_found", [])
        les = [x["size"] for x in found if x["strategy"] == "attacker-greedy"]
        solid = bool(a) and r.get("attack_grade") == "SOLID"
        rows.append(dict(fly=f, name=n, R1=W[n]["R1_strong_drivers"], R2=W[n]["R2_vertex_cut"],
                         cost=a["size"] if a else NONE, lesion=min(les) if les else NONE,
                         cost_solid=a["size"] if solid else NONE,
                         lesion_solid=(min(les) if les and solid and a["strategy"] == "attacker-greedy" else NONE)))


def rho(rs, x, y): return spearmanr([r[x] for r in rs], [r[y] for r in rs])[0] if len(rs) > 2 else float("nan")


def pooled(x, y, n_perm=10000, rng=np.random.default_rng(0)):
    """mean of per-fly Spearman rho (each fly's ranks separately), p = share of within-fly shuffles with mean >= observed"""
    groups = [[r for r in rows if r["fly"] == f] for f in FLIES]
    obs = np.nanmean([rho(g, x, y) for g in groups]); hit = 0
    for _ in range(n_perm):
        sh = [[dict(r, **{y: v}) for r, v in zip(g, rng.permutation([r[y] for r in g]))] for g in groups]
        hit += np.nanmean([rho(g, x, y) for g in sh]) >= obs
    return float(obs), (hit + 1) / (n_perm + 1)


report = {}
for label, c, l in (("primary", "cost", "lesion"), ("SOLID only", "cost_solid", "lesion_solid")):
    rep = {}
    for x, y in (("R1", c), ("R2", l)):
        per = {f: round(float(rho([r for r in rows if r["fly"] == f], x, y)), 3) for f in FLIES}
        p_rho, p = pooled(x, y); rep[f"{x} vs {y}"] = dict(per_fly=per, pooled_rho=round(p_rho, 3), p_one_sided=round(p, 4))
        print(f"[{label}] {x} vs {y}: per fly {per} | pooled rho {p_rho:.3f}, one-sided p {p:.4f}", flush=True)
    ok = all(v["pooled_rho"] >= 0.6 and v["p_one_sided"] < 0.05 for v in rep.values())
    rep["H1_confirmed"] = ok; report[label] = rep; print(f"[{label}] H1 {'CONFIRMED' if ok else 'NOT confirmed'}")
json.dump(dict(rows=rows, report=report), open(os.path.join(out, "h1_screen.json"), "w"), indent=1)
finish()
