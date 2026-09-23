"""The fly brain as a computer: connectome reservoir computing.

The idea (reservoir computing): leave the recurrent network completely untrained and train only a linear readout on
top of it. If the fixed wiring is rich enough, a linear readout of its activity can solve problems that the same
linear model cannot solve from the raw inputs. Nothing about the connectome is learned or tuned -- the only trained
parameters are the readout weights.

Setup:
  inputs   : 8 binary channels; each channel is a disjoint group of real sensory neurons, driven at 150 Hz for a 1 bit
             and 0 Hz for a 0 bit (the group is always present in the injection dict so the sub-network never changes).
  reservoir: the brain itself -- the LIF model of Shiu et al. 2024, unmodified.
  readout  : firing rates of a fixed set of neurons, fed to logistic regression (the only trained part).
  tasks    : IDENTITY (bit 0), AND, XOR (bits 0,1) and PARITY (bits 0,1,2). XOR and PARITY are NOT linearly separable
             from the raw bits, so a linear model on the inputs must fail at them. If it succeeds on the brain's
             activity, the brain did the nonlinear computation.
  controls : (a) logistic regression on the raw input bits -- the "no brain" baseline;
             (b) the same experiment on a SHUFFLED connectome (the same synapses redistributed among neurons,
                 preserving how many connections exist but destroying which neuron connects to which) -- the
                 "is the real wiring doing the work?" baseline.

Usage: python ml/reservoir.py --samples 300 --channels 8 --shuffled
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from brain import Brain

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"])
ap.add_argument("--samples", type=int, default=300, help="random input patterns presented to the brain")
ap.add_argument("--channels", type=int, default=8, help="binary input channels")
ap.add_argument("--per_channel", type=int, default=12, help="sensory neurons per channel")
ap.add_argument("--readout", type=int, default=200, help="neurons whose rates the linear readout sees")
ap.add_argument("--hz", type=float, default=150.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shuffled", action="store_true", help="also run the shuffled-connectome control")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
out_dir = os.path.join(HERE, "..", "atlas", args.sex, "ml"); os.makedirs(out_dir, exist_ok=True)

brain = Brain(args.sex)
sens = np.flatnonzero(np.isin([str(x) for x in brain.types["superclass"]], ["sensory", "sensory_ascending"]))
pick = rng.choice(sens, args.channels * args.per_channel, replace=False)
CHANNELS = [list(map(int, pick[i * args.per_channel:(i + 1) * args.per_channel])) for i in range(args.channels)]
ALL_IN = [i for ch in CHANNELS for i in ch]
print(f"{args.sex}: {args.channels} input channels x {args.per_channel} sensory neurons "
      f"({len(sens):,} sensory neurons available)", flush=True)


def present(bits, record):
    """Drive the channels whose bit is 1 and read the firing rates of `record`. Channels with a 0 bit stay in the
    injection dict at 0 Hz so that the sub-network is identical for every pattern."""
    inj = {i: (args.hz if bits[c] else 0.0) for c, ch in enumerate(CHANNELS) for i in ch}
    r = brain.run([], inject=inj, record=record, seed=1)
    return np.array([r["rates"][i] for i in record], np.float32), r


def collect(tag):
    """Present `samples` random bit patterns and return X (reservoir rates) and the bit matrix."""
    _, probe = present(np.ones(args.channels, int), [])                 # everything on: who responds at all?
    act = sorted(probe["active"], key=lambda i: -probe["active"][i])
    record = [int(i) for i in act if int(i) not in set(ALL_IN)][:args.readout]
    print(f"  [{tag}] {len(probe['active']):,} neurons respond; reading {len(record)} of them", flush=True)
    B = rng.integers(0, 2, (args.samples, args.channels))
    X = np.zeros((args.samples, len(record)), np.float32)
    t0 = time.time()
    for n in range(args.samples):
        X[n] = present(B[n], record)[0]
        if n % 50 == 0: print(f"    [{tag}] {n}/{args.samples}  {time.time()-t0:.0f}s", flush=True)
    return X, B, record


TASKS = {"IDENTITY (bit 0)": lambda B: B[:, 0],
         "AND (bits 0,1)": lambda B: B[:, 0] & B[:, 1],
         "XOR (bits 0,1)": lambda B: B[:, 0] ^ B[:, 1],
         "PARITY (bits 0,1,2)": lambda B: B[:, 0] ^ B[:, 1] ^ B[:, 2]}


def score(X, y):
    """5-fold cross-validated BALANCED accuracy of logistic regression (the only trained parameters)."""
    if len(set(y)) < 2: return float("nan")
    # Scale (rates are in Hz, so an unscaled penalty is meaningless) and let an INNER cross-validation pick the
    # regularisation strength; the outer 5-fold below never sees the C that was chosen for it.
    m = make_pipeline(StandardScaler(),
                      LogisticRegressionCV(Cs=np.logspace(-3, 3, 7), cv=4, max_iter=5000, scoring="balanced_accuracy"))
    # balanced accuracy: a finite sample of random bits is never exactly 50/50, and plain accuracy rewards a model
    # for guessing the majority class.
    return float(cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                 scoring="balanced_accuracy").mean())


results = {}
X, B, record = collect("real connectome")
Xs = None
if args.shuffled:
    W0 = brain.W.copy()
    coo = W0.tocoo()
    perm = rng.permutation(coo.data.size)                              # same synapses, redistributed among the pairs
    brain.W = sp.csr_matrix((coo.data, (coo.row[perm], coo.col[perm])), shape=W0.shape)
    brain._sub_cache.clear(); brain._run_cache.clear()
    Xs, Bs, _ = collect("shuffled connectome")
    brain.W = W0; brain._sub_cache.clear(); brain._run_cache.clear()

print(f"\n{'task':22s} {'raw bits':>10s} {'BRAIN':>10s}" + (f" {'shuffled':>10s}" if args.shuffled else ""))
for name, f in TASKS.items():
    y = f(B); raw, res = score(B.astype(float), y), score(X, y)
    row = {"raw_inputs": raw, "reservoir": res}
    line = f"{name:22s} {raw:9.2f} {res:10.2f}"
    if args.shuffled:
        row["shuffled"] = score(Xs, f(Bs)); line += f" {row['shuffled']:10.2f}"
    results[name] = row; print(line, flush=True)

print("\nchance is 0.50. 'raw bits' is a linear model on the inputs themselves: it can solve IDENTITY and AND, and")
print("provably cannot solve XOR or PARITY. Any accuracy the reservoir gains on those two is computation the fixed,")
print("untrained fly wiring performed -- the only trained parameters in the whole system are the readout weights.")
json.dump({"sex": args.sex, "samples": args.samples, "channels": args.channels, "per_channel": args.per_channel,
           "readout_neurons": len(record), "results": results}, open(os.path.join(out_dir, "reservoir.json"), "w"), indent=1)
np.savez_compressed(os.path.join(out_dir, "reservoir_data.npz"), X=X, B=B, record=np.array(record),
                    **({"Xs": Xs, "Bs": Bs} if args.shuffled else {}))
print(f"saved atlas/{args.sex}/ml/reservoir.json")
