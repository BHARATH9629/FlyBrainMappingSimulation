"""The fly brain as a computer: connectome reservoir computing.

The idea (reservoir computing): leave the recurrent network completely untrained and train only a linear readout on
top of it. If the fixed wiring is rich enough, a linear readout of its activity can solve problems that the same
linear model cannot solve from the raw inputs. Nothing about the connectome is learned or tuned -- the only trained
parameters are the readout weights.

Setup:
  inputs   : 8 binary channels; each channel is a disjoint group of real sensory neurons, driven at 150 Hz for a 1 bit
             and 0 Hz for a 0 bit (the group is always present in the injection dict so the sub-network never changes).
  reservoir: the brain itself -- the LIF model of Shiu et al. 2024, unmodified. Every trial gets its own random seed,
             so the readout must work through the model's Poisson noise.
  readout  : firing rates of the K most responsive non-input neurons, fed to logistic regression (the only trained
             part), for several K.
  tasks    : IDENTITY (bit 0), AND, XOR (bits 0,1) and PARITY (bits 0,1,2). XOR and PARITY are NOT linearly separable
             from the raw bits, so a linear model on the inputs must fail at them. If it succeeds on the brain's
             activity, the brain did the nonlinear computation.
  controls : (a) logistic regression on the raw input bits -- the "no brain" baseline;
             (b) a REWIRED connectome: every neuron keeps its outgoing synapses (so its sign, i.e. its transmitter,
                 and its total output) and every neuron keeps its number of incoming connections, but who connects to
                 whom is randomised (the targets of all connections are permuted). The "is the real wiring doing the
                 work?" baseline. Both brains are read with the SAME number of neurons K.
  repeats  : --draws independent choices of the input neurons; results are reported as mean +- sd over draws.

Usage: python ml/reservoir.py --samples 300 --draws 3 --shuffled
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from brain import Brain

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"])
ap.add_argument("--samples", type=int, default=300, help="random input patterns presented to the brain, per draw")
ap.add_argument("--channels", type=int, default=8, help="binary input channels")
ap.add_argument("--per_channel", type=int, default=12, help="sensory neurons per channel")
ap.add_argument("--readouts", default="25,50,100,200", help="readout sizes K compared (same K for both brains)")
ap.add_argument("--draws", type=int, default=3, help="independent random choices of the input neurons")
ap.add_argument("--hz", type=float, default=150.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shuffled", action="store_true", help="also run the rewired-connectome control")
ap.add_argument("--workers", type=int, default=6, help="trials run in parallel on this many CPU cores")
args = ap.parse_args() if __name__ == "__main__" else ap.parse_args([])
KS = [int(k) for k in args.readouts.split(",")]
out_dir = os.path.join(HERE, "..", "atlas", args.sex, "ml"); os.makedirs(out_dir, exist_ok=True)

brain = None                                    # the main process's Brain (loaded in main(); workers load their own)
TASKS = {"IDENTITY (bit 0)": lambda B: B[:, 0],
         "AND (bits 0,1)": lambda B: B[:, 0] & B[:, 1],
         "XOR (bits 0,1)": lambda B: B[:, 0] ^ B[:, 1],
         "PARITY (bits 0,1,2)": lambda B: B[:, 0] ^ B[:, 1] ^ B[:, 2]}


_WB = {}                                        # worker process: its own Brain, real or rewired


def _init(sex, rewire_seed):
    b = Brain(sex)
    if rewire_seed is not None: b.W = rewired(b.W, np.random.default_rng(rewire_seed))
    _WB["b"] = b


def _trial(job):
    inj, record, seed = job
    r = _WB["b"].run([], inject=inj, record=record, seed=seed)
    return np.array([r["rates"][i] for i in record], np.float32)


def rewired(W, rng):
    """Degree-preserving rewiring: every connection keeps its presynaptic neuron and its signed weight (so each neuron
    keeps its out-degree, total output and transmitter sign), and the postsynaptic targets are permuted across all
    connections (so each neuron keeps its in-degree). Only who-connects-to-whom is randomised."""
    c = W.tocoo()
    return sp.csr_matrix((c.data, (c.row, c.col[rng.permutation(c.col.size)])), shape=W.shape)   # duplicates are summed


def present(channels, bits, record, seed):
    """Drive the channels whose bit is 1 and read the firing rates of `record`. Channels with a 0 bit stay in the
    injection dict at 0 Hz so that the sub-network is identical for every pattern."""
    inj = {i: (args.hz if bits[c] else 0.0) for c, ch in enumerate(channels) for i in ch}
    r = brain.run([], inject=inj, record=record, seed=seed)
    return np.array([r["rates"][i] for i in record], np.float32), r


def collect(tag, channels, B, seeds, rewire_seed=None):
    """Present the bit patterns B and return X (rates of up to max(KS) readout neurons, most responsive first)."""
    all_in = {i for ch in channels for i in ch}
    _, probe = present(channels, np.ones(args.channels, int), [], seed=10_000)       # everything on: who responds?
    act = sorted(probe["active"], key=lambda i: -probe["active"][i])
    record = [int(i) for i in act if int(i) not in all_in][:max(KS)]
    print(f"  [{tag}] {len(probe['active']):,} neurons respond; recording {len(record)}", flush=True)
    jobs = [({i: (args.hz if B[n][c] else 0.0) for c, ch in enumerate(channels) for i in ch}, record, int(seeds[n]))
            for n in range(len(B))]
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(args.workers, initializer=_init, initargs=(args.sex, rewire_seed)) as pool:   # the same brain in every worker
        X = np.array(pool.map(_trial, jobs, chunksize=4), np.float32)
    print(f"    [{tag}] {len(B)} trials in {time.time()-t0:.0f}s", flush=True)
    return X


def score(X, y):
    """5-fold cross-validated BALANCED accuracy of logistic regression (the only trained parameters)."""
    if len(set(y)) < 2 or X.shape[1] == 0: return float("nan")
    # Scale (rates are in Hz, so an unscaled penalty is meaningless) and let an INNER cross-validation pick the
    # regularisation strength; the outer 5-fold below never sees the C that was chosen for it.
    m = make_pipeline(StandardScaler(),
                      LogisticRegressionCV(Cs=np.logspace(-3, 3, 7), cv=4, max_iter=5000, scoring="balanced_accuracy"))
    # balanced accuracy: a finite sample of random bits is never exactly 50/50, and plain accuracy rewards a model
    # for guessing the majority class.
    return float(cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                 scoring="balanced_accuracy").mean())


def main():
    global brain
    brain = Brain(args.sex)
    SENS = np.flatnonzero(np.isin([str(x) for x in brain.types["superclass"]], ["sensory", "sensory_ascending"]))
    W0 = brain.W.copy(); rows = []                     # rows: (draw, brain, K, task, balanced accuracy)
    for d in range(args.draws):
        rng = np.random.default_rng(args.seed + d)
        pick = rng.choice(SENS, args.channels * args.per_channel, replace=False)
        channels = [list(map(int, pick[i * args.per_channel:(i + 1) * args.per_channel])) for i in range(args.channels)]
        B = rng.integers(0, 2, (args.samples, args.channels)); seeds = rng.integers(1, 10**9, args.samples)
        print(f"draw {d + 1}/{args.draws}: {args.channels} channels x {args.per_channel} sensory neurons", flush=True)
        for name, f in TASKS.items(): rows.append((d, "raw inputs", 0, name, score(B.astype(float), f(B))))
        rw_seed = args.seed * 1000 + d + 1              # the rewired brain of this draw, rebuilt identically in every worker
        brains = [("real", None)] + ([("rewired", rw_seed)] if args.shuffled else [])
        for tag, rs in brains:
            brain.W = W0 if rs is None else rewired(W0, np.random.default_rng(rs)); brain._sub_cache.clear(); brain._run_cache.clear()
            X = collect(f"{tag} connectome", channels, B, seeds, rewire_seed=rs)
            for K in KS:
                for name, f in TASKS.items():
                    rows.append((d, tag, K, name, score(X[:, :K], f(B)) if X.shape[1] >= K else float("nan")))
        brain.W = W0; brain._sub_cache.clear(); brain._run_cache.clear()


    def summary(tag, K, task):
        v = np.array([r[4] for r in rows if r[1] == tag and r[2] == K and r[3] == task], float)
        v = v[np.isfinite(v)]
        return (float(v.mean()), float(v.std()), int(v.size)) if v.size else (float("nan"), float("nan"), 0)


    print(f"\nbalanced accuracy, mean +- sd over {args.draws} draws of the input neurons (chance = 0.50)")
    hdr = f"{'task':22s} {'raw bits':>12s}" + "".join(f" {'real K=' + str(K):>13s}" + (f" {'rewired K=' + str(K):>13s}" if args.shuffled else "") for K in KS)
    print(hdr)
    for task in TASKS:
        m, s, _ = summary("raw inputs", 0, task); line = f"{task:22s} {m:6.2f}+-{s:4.2f}"
        for K in KS:
            m, s, n = summary("real", K, task); line += f"   {m:6.2f}+-{s:4.2f}" if n else f"   {'n/a':>11s}"
            if args.shuffled:
                m, s, n = summary("rewired", K, task); line += f"   {m:6.2f}+-{s:4.2f}" if n else f"   {'n/a':>11s}"
        print(line, flush=True)
    print("\n'raw bits' = a linear model on the inputs themselves: it can solve IDENTITY and AND and provably cannot solve")
    print("XOR or PARITY. 'rewired' = same neurons, same signs, same numbers of connections, random partners.")
    print("n/a = the rewired brain has fewer than K responding neurons (both brains are always read with the same K).")
    json.dump({"sex": args.sex, "samples": args.samples, "channels": args.channels, "per_channel": args.per_channel,
               "draws": args.draws, "readouts": KS, "trial_noise": "own seed per trial",
               "control": "degree-preserving rewiring (targets permuted; presynaptic neuron, weight and sign kept)",
               "rows": [dict(draw=r[0], brain=r[1], K=r[2], task=r[3], balanced_accuracy=r[4]) for r in rows]},
              open(os.path.join(out_dir, "reservoir.json"), "w"), indent=1)
    print(f"saved atlas/{args.sex}/ml/reservoir.json")


if __name__ == "__main__":
    main()
