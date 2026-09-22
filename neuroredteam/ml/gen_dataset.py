"""Generate the intervention -> readout dataset for the surrogate brain.

Pool = neurons active with the stimulus + the readout's strongest inputs (excitatory and inhibitory).
Samples = every single-neuron lesion / inject / weaken in the pool (the per-neuron labels), plus random mixed sets
of size 2-8, plus the atlas's own attacks and repairs. Each sample: MN9 (readout) rate with and without the stimulus.
Writes atlas/<sex>/ml/dataset_<behaviour>.npz  (X: [n_samples, 3 * pool] = lesion flag, inject rate/150, weaken factor;
y_on, y_off; pool indices; meta) and pool_<behaviour>.csv (per-neuron wiring features).
Usage: python ml/gen_dataset.py --sex female --behaviour eat --n_random 4000
"""
import argparse, json, os, sys, time, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from brain import Brain
from behaviours import Behaviour, RULES

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--n_random", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pool_max", type=int, default=600)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); ro = b.readout; W = brain.W
out_dir = os.path.join(HERE, "..", "atlas", args.sex, "ml"); os.makedirs(out_dir, exist_ok=True)
t0 = time.time()

# ---------------------------------------------------------------- the pool
_, r0 = b.measure(True)
col = W[:, ro].toarray().ravel()
active = {int(i): hz for i, hz in r0["active"].items() if i not in b.protected}
inputs = [int(i) for i in np.argsort(-np.abs(col)) if col[i] != 0 and i not in b.protected][:80]
pool = sorted(set(active) | set(inputs))
if len(pool) > args.pool_max:                              # keep the most active + all direct inputs
    keep = set(inputs) | set(sorted(active, key=lambda i: -active[i])[:args.pool_max - len(inputs)])
    pool = sorted(keep)
P = len(pool); loc = {g: k for k, g in enumerate(pool)}
print(f"{args.sex} {b.name}: pool of {P} neurons ({len(active)} active with stimulus, {len(inputs)} direct inputs)", flush=True)

# per-neuron wiring features (for the choke-point predictor)
stim_drive = np.asarray(abs(W[b.stim]).sum(axis=0)).ravel()
outdeg = np.asarray((abs(W) > 0).sum(axis=1)).ravel(); indeg = np.asarray((abs(W) > 0).sum(axis=0)).ravel()
outstr = np.asarray(abs(W).sum(axis=1)).ravel(); instr = np.asarray(abs(W).sum(axis=0)).ravel()
exc_out = np.asarray(W.maximum(0).sum(axis=1)).ravel(); inh_out = np.asarray((-W).maximum(0).sum(axis=1)).ravel()
# 2-hop: synapses onto the readout's direct excitatory drivers / inhibitors
drivers = [int(i) for i in np.flatnonzero(col > 0)]; inhib = [int(i) for i in np.flatnonzero(col < 0)]
to_drivers = np.asarray(abs(W[:, drivers]).sum(axis=1)).ravel() if drivers else np.zeros(brain.N)
to_inhib = np.asarray(abs(W[:, inhib]).sum(axis=1)).ravel() if inhib else np.zeros(brain.N)
T = brain.types
with open(os.path.join(out_dir, f"pool_{b.name}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "id", "type", "superclass", "class", "side", "rate_hz", "syn_to_readout", "stim_drive", "in_deg", "out_deg",
                "in_str", "out_str", "exc_out", "inh_out", "to_drivers", "to_inhibitors", "is_inhibitory"])
    for i in pool:
        w.writerow([i, int(brain.ids[i]), brain.type_of(i), T["superclass"].iloc[i], T["class"].iloc[i], T["side"].iloc[i],
                    round(active.get(i, 0.0), 1), float(col[i]), float(stim_drive[i]), int(indeg[i]), int(outdeg[i]), float(instr[i]),
                    float(outstr[i]), float(exc_out[i]), float(inh_out[i]), float(to_drivers[i]), float(to_inhib[i]), int(inh_out[i] > exc_out[i])])

# ---------------------------------------------------------------- samples
HZ = RULES["moves"]["inject"]["max_hz"]; WK = RULES["moves"]["weaken"]["min_factor"]
samples = []                                                # list of intervention dicts (global indices)
for i in pool:                                              # every single move
    samples.append({"lesion": [i]}); samples.append({"inject": {i: HZ}}); samples.append({"weaken": {i: WK}})
for _ in range(args.n_random):                              # random mixed sets
    k = int(rng.integers(2, 9)); chosen = rng.choice(pool, k, replace=False); iv = {}
    for i in chosen:
        m = rng.choice(["lesion", "inject", "weaken"], p=[0.5, 0.3, 0.2])
        if m == "lesion": iv.setdefault("lesion", []).append(int(i))
        elif m == "inject": iv.setdefault("inject", {})[int(i)] = float(rng.choice([50, 100, HZ]))
        else: iv.setdefault("weaken", {})[int(i)] = float(rng.choice([0.25, 0.5, 0.75]))
    samples.append(iv)
# the agents' own experiments (attacks, repairs, probes) from the atlas logs -- attacker moves only
for fn in os.listdir(os.path.join(HERE, "..", "atlas", args.sex)):
    if fn.endswith(".json") and b.name in fn:
        try: d = json.load(open(os.path.join(HERE, "..", "atlas", args.sex, fn)))
        except Exception: continue
        if not isinstance(d, dict) or not isinstance(d.get("entry"), dict): continue      # logs / chokepoint files
        for g in (d["entry"].get("goals", {}) or {}).values():
            for atk in (g.get("attacks") or {}).values():
                if atk and atk.get("intervention"):
                    iv = {k: v for k, v in atk["intervention"].items() if k in ("lesion", "inject", "weaken")}
                    iv = {k: ([int(x) for x in v] if k == "lesion" else {int(a): float(c) for a, c in v.items()}) for k, v in iv.items()}
                    if iv and all(i in loc for i in (set(iv.get("lesion", [])) | set(iv.get("inject", {})) | set(iv.get("weaken", {})))): samples.append(iv)
print(f"{len(samples)} samples ({3 * P} single moves, {args.n_random} random sets, {len(samples) - 3 * P - args.n_random} from the atlas)", flush=True)


def encode(iv):
    x = np.zeros(3 * P, np.float32)
    for i in iv.get("lesion", []): x[loc[i]] = 1.0
    for i, hz in iv.get("inject", {}).items(): x[P + loc[i]] = hz / HZ
    for i, f in iv.get("weaken", {}).items(): x[2 * P + loc[i]] = 1.0 - f       # 0 = untouched, 0.5 = halved
    return x


X = np.zeros((len(samples), 3 * P), np.float32); y_on = np.zeros(len(samples), np.float32); y_off = np.zeros(len(samples), np.float32)
for n, iv in enumerate(samples):
    X[n] = encode(iv)
    y_on[n] = b.measure(True, seed=1, **iv)[0]; y_off[n] = b.measure(False, seed=1, **iv)[0]
    if n % 200 == 0:
        print(f"  {n}/{len(samples)}  {time.time() - t0:.0f}s  last on {y_on[n]:.0f} / off {y_off[n]:.0f} Hz", flush=True)
        np.savez_compressed(os.path.join(out_dir, f"dataset_{b.name}.npz"), X=X[:n + 1], y_on=y_on[:n + 1], y_off=y_off[:n + 1], pool=np.array(pool), healthy=r0["rates"][ro])
np.savez_compressed(os.path.join(out_dir, f"dataset_{b.name}.npz"), X=X, y_on=y_on, y_off=y_off, pool=np.array(pool), healthy=r0["rates"][ro])
print(f"saved {out_dir}/dataset_{b.name}.npz: {len(samples)} samples, pool {P}  ({time.time() - t0:.0f}s)")
