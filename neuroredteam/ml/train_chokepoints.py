"""Choke-point predictor: can a neuron's causal importance be predicted from its wiring alone?

Label per pool neuron (from the single-move samples of the dataset): the drop in readout rate when that neuron is
lesioned alone, and when it is injected alone (Hz). Features: wiring statistics from pool_<behaviour>.csv plus cell
class. Models: gradient boosting with 5-fold cross-validation; baselines: each single feature (Spearman).
Also: classification of 'in a minimal attack' (from chokepoints_<behaviour>.json) with cross-validated AUC.
Usage: python ml/train_chokepoints.py --sex female --behaviour eat
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
args = ap.parse_args()
d = os.path.join(HERE, "..", "atlas", args.sex, "ml")
D = np.load(os.path.join(d, f"dataset_{args.behaviour}.npz")); X, y_on, pool = D["X"], D["y_on"], D["pool"]; P = len(pool)
healthy = float(D["healthy"])
feat = pd.read_csv(os.path.join(d, f"pool_{args.behaviour}.csv")).set_index("index").loc[pool]

# labels from the single-move rows: gen_dataset writes, per pool neuron k, rows 3k = lesion, 3k+1 = inject, 3k+2 = weaken
L, I = np.arange(0, 3 * P, 3), np.arange(1, 3 * P, 3)
assert all(X[L[k], k] == 1.0 and X[I[k], P + k] == 1.0 for k in range(P)), "unexpected single-move layout"
drop_lesion = healthy - y_on[L]                  # Hz lost when this neuron is cut
drop_inject = healthy - y_on[I]                  # Hz lost when this neuron is driven at 150 Hz (inhibitors) -- can be negative
feat["drop_lesion"] = drop_lesion; feat["drop_inject"] = drop_inject
num = ["rate_hz", "syn_to_readout", "stim_drive", "in_deg", "out_deg", "in_str", "out_str", "exc_out", "inh_out", "to_drivers", "to_inhibitors", "is_inhibitory"]
F = feat[num].copy(); F["abs_syn_to_readout"] = F["syn_to_readout"].abs()
for c in ("superclass", "class"):
    for v in feat[c].astype(str).unique(): F[f"{c}={v}"] = (feat[c].astype(str) == v).astype(int)
Fv = F.values.astype(float)

print(f"{args.sex} {args.behaviour}: {P} pool neurons; healthy {healthy:.0f} Hz")
print(f"  single-cut effect: {int((drop_lesion >= 20).sum())} neurons drop MN9 by >= 20 Hz, {int((drop_lesion >= healthy - 10).sum())} block it alone")
print(f"  single-inject effect: {int((drop_inject >= 20).sum())} neurons drop MN9 by >= 20 Hz when driven")

for target, y in (("drop when cut", drop_lesion), ("drop when driven", drop_inject)):
    print(f"\n== predicting '{target}' from wiring ==")
    print("  single-feature Spearman: " + ", ".join(f"{c} {spearmanr(F[c], y)[0]:+.2f}" for c in ("abs_syn_to_readout", "rate_hz", "stim_drive", "to_drivers", "out_str", "is_inhibitory")))
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    pred = cross_val_predict(gb, Fv, y, cv=KFold(5, shuffle=True, random_state=0))
    rho = spearmanr(pred, y)[0]; mae = np.abs(pred - y).mean()
    print(f"  gradient boosting (5-fold CV): Spearman {rho:+.2f}, MAE {mae:.1f} Hz")
    top = np.argsort(-y)[:10]; hit = len(set(top) & set(np.argsort(-pred)[:10]))
    print(f"  of the 10 most important neurons, the model puts {hit} in its own top 10")

# classification: in any minimal attack found by chokepoints.py
cp = os.path.join(HERE, "..", "atlas", args.sex, f"chokepoints_{args.behaviour}.json")
if os.path.exists(cp):
    c = json.load(open(cp)); members = set(i for S in c["attacks"] for i in S)
    lab = np.array([1 if int(i) in members else 0 for i in pool])
    print(f"\n== classifying 'appears in a minimal attack' ({lab.sum()} positives of {P}) ==")
    if lab.sum() >= 5:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, class_weight="balanced", random_state=0)
        prob = cross_val_predict(clf, Fv, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print(f"  wiring only:            AUC {roc_auc_score(lab, prob):.2f}, average precision {average_precision_score(lab, prob):.2f} (chance {lab.mean():.2f})")
        Fv2 = np.column_stack([Fv, drop_lesion, drop_inject])
        prob2 = cross_val_predict(clf, Fv2, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print(f"  wiring + single-move effects: AUC {roc_auc_score(lab, prob2):.2f}, average precision {average_precision_score(lab, prob2):.2f}")
    else: print("  too few positives to classify")
feat.to_csv(os.path.join(d, f"neuron_effects_{args.behaviour}.csv"))
print(f"\nsaved {d}/neuron_effects_{args.behaviour}.csv (per-neuron wiring features + measured single-move effects)")
