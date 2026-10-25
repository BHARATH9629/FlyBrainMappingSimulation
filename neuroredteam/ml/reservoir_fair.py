"""Connectome reservoir computing, the fair version: can the untrained fly wiring compute XOR / PARITY for a linear
readout, on input patterns the readout has NEVER seen, through the model's own trial-to-trial noise?

Differences from ml/reservoir.py:
  * listening window: every trial is simulated for 1000 ms and read out over the first 300 ms AND the whole 1000 ms
    (the same trials, so the window is the only difference);
  * inputs: two modes -- 'neurons' (each channel = 12 random sensory neurons, as before) and 'types' (each channel =
    all neurons of one real sensory cell type, 8-40 neurons);
  * readout: a pool of the 500 most active non-input neurons (chosen from an all-inputs-on probe, which never sees the
    labels); inside every training fold the K most informative of them are selected (univariate F test) and a
    logistic regression is trained -- the test fold never influences the selection;
  * testing: StratifiedGroupKFold grouped by the full 8-bit pattern, so every test pattern is one the readout never
    saw in training (no memorising);
  * control: the degree-preserving rewired connectome of ml/reservoir.py, same K.

Usage: python ml/reservoir_fair.py --samples 240 --draws 3
"""
import argparse, json, os, sys, time, warnings
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from brain import Brain
from reservoir import rewired

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"])
ap.add_argument("--samples", type=int, default=240); ap.add_argument("--draws", type=int, default=3)
ap.add_argument("--channels", type=int, default=8); ap.add_argument("--per_channel", type=int, default=12)
ap.add_argument("--modes", default="neurons,types"); ap.add_argument("--ks", default="10,25,50")
ap.add_argument("--windows", default="300,1000", help="read-out windows in ms (the trial lasts the longest one)")
ap.add_argument("--pool", type=int, default=500); ap.add_argument("--hz", type=float, default=150.0)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args() if __name__ == "__main__" else ap.parse_args([])
KS = [int(k) for k in args.ks.split(",")]; WINS = [int(w) for w in args.windows.split(",")]; T_MS = max(WINS)
TASKS = {"IDENTITY (bit 0)": lambda B: B[:, 0], "AND (bits 0,1)": lambda B: B[:, 0] & B[:, 1],
         "XOR (bits 0,1)": lambda B: B[:, 0] ^ B[:, 1], "PARITY (bits 0,1,2)": lambda B: B[:, 0] ^ B[:, 1] ^ B[:, 2]}
_WB = {}


def _init(sex, rewire_seed, wins):
    _WB["wins"] = wins                              # passed in: a spawned worker does not see the command line
    b = Brain(sex)
    if rewire_seed is not None: b.W = rewired(b.W, np.random.default_rng(rewire_seed))
    _WB["b"] = b


def _trial(job):
    """one trial, T_MS long: the rate of every pool neuron over each read-out window"""
    inj, record, seed = job; wins = _WB["wins"]
    r = _WB["b"].run([], inject=inj, record=record, seed=seed, t_ms=max(wins), spikes=True)
    pos = {g: k for k, g in enumerate(record)}
    m = np.isin(r["spike_i"], record); st, si = r["spike_t"][m], r["spike_i"][m]
    col = np.array([pos[int(g)] for g in si], int)
    return np.stack([np.bincount(col[st < w], minlength=len(record)) * (1000.0 / w) for w in wins]).astype(np.float32)


def channels_for(brain, mode, rng):
    sens_mask = np.isin([str(x) for x in brain.types["superclass"]], ["sensory", "sensory_ascending"])
    if mode == "neurons":
        pick = rng.choice(np.flatnonzero(sens_mask), args.channels * args.per_channel, replace=False)
        return [list(map(int, pick[i * args.per_channel:(i + 1) * args.per_channel])) for i in range(args.channels)], None
    T = np.array([str(x) for x in brain.types["type"]])
    types, counts = np.unique(T[sens_mask], return_counts=True)
    ok = [t for t, c in zip(types, counts) if 8 <= c <= 40 and t not in ("nan", "None")]
    chosen = list(rng.choice(ok, args.channels, replace=False))
    return [list(map(int, np.flatnonzero(sens_mask & (T == t)))) for t in chosen], chosen


def collect(brain, channels, B, seeds, rewire_seed):
    """X[window] = (trials x pool) rates; the pool comes from an all-on probe (never from the labels)"""
    all_in = {i for ch in channels for i in ch}
    inj_on = {i: args.hz for ch in channels for i in ch}
    probe = brain.run([], inject=inj_on, record=[], seed=10_000)
    act = sorted(probe["active"], key=lambda i: -probe["active"][i])
    pool = [int(i) for i in act if int(i) not in all_in][:args.pool]
    jobs = [({i: (args.hz if B[n][c] else 0.0) for c, ch in enumerate(channels) for i in ch}, pool, int(seeds[n])) for n in range(len(B))]
    from multiprocessing import Pool
    with Pool(args.workers, initializer=_init, initargs=(args.sex, rewire_seed, WINS)) as p:
        R = np.array(p.map(_trial, jobs, chunksize=2), np.float32)          # trials x windows x pool
    return {w: R[:, k, :] for k, w in enumerate(WINS)}, len(probe["active"]), len(pool)


def score(X, y, groups, K):
    if len(set(y)) < 2 or X.shape[1] < K: return float("nan")
    m = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=K),
                      LogisticRegressionCV(Cs=np.logspace(-3, 3, 7), cv=3, max_iter=5000, scoring="balanced_accuracy"))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(m, np.nan_to_num(X), y, groups=groups, cv=cv, scoring="balanced_accuracy").mean())


def main():
    brain = Brain(args.sex); W0 = brain.W.copy(); rows = []; meta = []
    out_dir = os.path.join(HERE, "..", "atlas", args.sex, "ml"); t_all = time.time()
    for mode in args.modes.split(","):
        for d in range(args.draws):
            rng = np.random.default_rng(args.seed + 100 * d + (0 if mode == "neurons" else 7))
            channels, types = channels_for(brain, mode, rng)
            B = rng.integers(0, 2, (args.samples, args.channels)); seeds = rng.integers(1, 10**9, args.samples)
            groups = B @ (1 << np.arange(args.channels))
            print(f"[{mode} draw {d + 1}] channels: {types or [len(c) for c in channels]}", flush=True)
            for name, f in TASKS.items():                   # the "no brain" baseline: the same linear readout on the bits
                rows.append(dict(mode=mode, draw=d, brain="raw inputs", window=0, K=0, task=name,
                                 acc=score(B.astype(float), f(B), groups, args.channels)))
            for tag, rs in (("real", None), ("rewired", args.seed * 1000 + 100 * d + 1 + (0 if mode == "neurons" else 50))):
                brain.W = W0 if rs is None else rewired(W0, np.random.default_rng(rs)); brain._sub_cache.clear(); brain._run_cache.clear()
                t0 = time.time(); Xw, n_resp, n_pool = collect(brain, channels, B, seeds, rs)
                print(f"   {tag:8s}: {n_resp:,} neurons respond, pool {n_pool}, {len(B)} trials in {time.time() - t0:.0f}s", flush=True)
                meta.append(dict(mode=mode, draw=d, brain=tag, responding=n_resp, pool=n_pool))
                for w in WINS:
                    for K in KS:
                        for name, f in TASKS.items():
                            rows.append(dict(mode=mode, draw=d, brain=tag, window=w, K=K, task=name, acc=score(Xw[w], f(B), groups, K)))
            brain.W = W0; brain._sub_cache.clear(); brain._run_cache.clear()
            json.dump(dict(args=vars(args), rows=rows, meta=meta), open(os.path.join(out_dir, "reservoir_fair.json"), "w"), indent=1)

    def agg(**q):
        v = np.array([r["acc"] for r in rows if all(r[k] == x for k, x in q.items())], float); v = v[np.isfinite(v)]
        return f"{v.mean():.2f}+-{v.std():.2f}" if v.size else "   n/a   "
    for mode in args.modes.split(","):
        print(f"\n=== inputs: {mode}   (balanced accuracy on UNSEEN input patterns, mean +- sd over {args.draws} draws; chance 0.50)")
        print(f"{'task':20s} {'raw bits':>10s}" + "".join(f" {tag[:4] + ' ' + str(w) + 'ms K' + str(K):>17s}" for w in WINS for K in KS for tag in ("real", "rewired")))
        for name in TASKS:
            print(f"{name:20s} {agg(mode=mode, brain='raw inputs', task=name):>10s}" +
                  "".join(f" {agg(mode=mode, brain=tag, window=w, K=K, task=name):>17s}" for w in WINS for K in KS for tag in ("real", "rewired")), flush=True)
    for m_ in meta: print(f"  {m_['mode']:8s} draw {m_['draw'] + 1} {m_['brain']:8s}: {m_['responding']:,} neurons respond")
    print(f"\ntotal {time.time() - t_all:.0f}s; saved atlas/{args.sex}/ml/reservoir_fair.json")


if __name__ == "__main__":
    main()
