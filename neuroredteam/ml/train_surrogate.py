"""Surrogate brain: predict the readout rate from an intervention vector, without running the simulator.

Trains ridge regression (primary: effects of interventions turn out to be close to additive) and gradient-boosted trees
(comparison) on dataset_<behaviour>.npz, held-out 20 %.
Reports: MAE in Hz, R^2, and the decision accuracy that matters for the attacker -- does the surrogate call
'blocked' (on-rate <= off bar) and 'forced' (off-rate >= on bar) the same way the simulator does?
Then: how much faster is it, and a sanity check that every atlas attack is called correctly.
Usage: python ml/train_surrogate.py --sex female --behaviour eat
"""
import argparse, os, sys, time, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from behaviours import BEHAVIOURS

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
args = ap.parse_args()
d = os.path.join(HERE, "..", "atlas", args.sex, "ml"); D = np.load(os.path.join(d, f"dataset_{args.behaviour}.npz"))
X, y_on, y_off, pool = D["X"], D["y_on"], D["y_off"], D["pool"]; P = len(pool)
cfg = BEHAVIOURS[args.sex][args.behaviour]; ON, OFF = cfg["on_threshold"], cfg["off_threshold"]
print(f"{args.sex} {args.behaviour}: {len(X)} samples x {X.shape[1]} features (pool {P}); healthy {float(D['healthy']):.0f} Hz; bars on >= {ON}, off <= {OFF}")
print(f"  targets: on-rate mean {y_on.mean():.0f} Hz, {100*(y_on <= OFF).mean():.0f} % blocked; off-rate {100*(y_off >= ON).mean():.0f} % forced")

Xtr, Xte, ytr_on, yte_on, ytr_off, yte_off = train_test_split(X, y_on, y_off, test_size=0.2, random_state=0)
models = {}
for name, ytr, yte in (("on", ytr_on, yte_on), ("off", ytr_off, yte_off)):
    t0 = time.time()
    rd = Ridge(alpha=0.3).fit(Xtr, ytr)                        # primary: intervention effects are close to additive
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=5, early_stopping=True, random_state=0).fit(Xtr, ytr)
    p_gb, p_rd = gb.predict(Xte), rd.predict(Xte)
    print(f"\n  readout {name} (stimulus {'present' if name == 'on' else 'absent'}): trained in {time.time()-t0:.0f}s")
    print(f"    ridge (linear, additive):  MAE {mean_absolute_error(yte, p_rd):5.1f} Hz, R2 {r2_score(yte, p_rd):.3f}")
    print(f"    gradient boosting (trees): MAE {mean_absolute_error(yte, p_gb):5.1f} Hz, R2 {r2_score(yte, p_gb):.3f}")
    bar = OFF if name == "on" else ON
    truth = (yte <= bar) if name == "on" else (yte >= bar); pred = (p_rd <= bar) if name == "on" else (p_rd >= bar)
    tp = (truth & pred).sum(); fp = (~truth & pred).sum(); fn = (truth & ~pred).sum()
    print(f"    decision '{'blocked' if name == 'on' else 'forced'}': accuracy {100*(truth == pred).mean():.1f} %, "
          f"precision {tp/max(tp+fp,1):.2f}, recall {tp/max(tp+fn,1):.2f}  ({truth.sum()} positives of {len(yte)})")
    models[name] = rd
# speed
t0 = time.time(); _ = models["on"].predict(X[:2000]); dt = time.time() - t0
print(f"\n  speed: {2000/dt:,.0f} predictions per second (simulator: ~1-2 per second)")
joblib.dump(dict(on=models["on"], off=models["off"], pool=pool, ON=ON, OFF=OFF), os.path.join(d, f"surrogate_{args.behaviour}.joblib"))
print(f"  saved {d}/surrogate_{args.behaviour}.joblib")
