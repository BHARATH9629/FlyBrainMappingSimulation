"""Control B analysis (pre-registration log, 27 Sep + clarification): for each model variant, Spearman correlation
between baseline and variant attack cost over the 16 pre-drawn fly-runs (no attack found = ranked above every found
cost). Runs failing the 8-seed inclusion check under a variant are dropped from that variant (and counted); a variant
with fewer than 8 remaining is 'not assessable'. Holds if rho >= 0.7 for every assessable variant."""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr
from guard import start, finish
out = start("robust")
R = os.path.dirname(out); SUB = json.load(open(os.path.join(out, "subset_B.json"))); NONE = 10 ** 3; rep = {}
base = {n: json.load(open(os.path.join(R, "instances", f"{n}_block.json")))["attack"]["size"] for n in SUB}
for d in sorted(glob.glob(os.path.join(R, "robust_B_*"))):
    var = os.path.basename(d).replace("robust_B_", ""); rows = []; missing = []
    for n in SUB:
        fn = os.path.join(d, f"{n}_block.json")
        if not os.path.exists(fn): missing.append(n); continue
        r = json.load(open(fn))
        rows.append(dict(name=n, included=bool(r.get("included", True)), base=base[n],
                         variant=(r["attack"]["size"] if r.get("attack") else NONE) if r.get("included", True) else None,
                         grade=r.get("attack_grade")))
    inc = [r for r in rows if r["included"]]
    if missing: status, rho, p = f"INCOMPLETE ({len(missing)} missing)", None, None
    elif len(inc) < 8: status, rho, p = "NOT ASSESSABLE (< 8 included)", None, None
    else:
        rho, p = spearmanr([r["base"] for r in inc], [r["variant"] for r in inc]); rho, p = float(rho), float(p)
        status = "HOLDS" if rho >= 0.7 else "FAILS"
    found = [r for r in inc if r["variant"] != NONE]
    rep[var] = dict(status=status, rho=rho, p=p, n_included=len(inc), n_excluded=len(rows) - len(inc), missing=missing,
                    break_rate=(len(found) / len(inc)) if inc else None, baseline_break_rate=1.0, rows=rows)
    print(f"{var:16s} included {len(inc):2d}/16  broken {len(found):2d}  rho {rho if rho is None else round(rho, 3)}  -> {status}")
ok = all(v["status"] in ("HOLDS", "NOT ASSESSABLE (< 8 included)") for v in rep.values()) and any(v["status"] == "HOLDS" for v in rep.values())
rep["control_B_holds"] = bool(ok); print("control B", "HOLDS" if ok else "does NOT hold")
json.dump(rep, open(os.path.join(out, "control_B.json"), "w"), indent=1)
finish()
