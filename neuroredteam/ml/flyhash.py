"""FlyHash with the REAL fly wiring: the projection-neuron -> Kenyon-cell connectome used as an AI layer.

The fly olfactory circuit maps ~50 input channels (glomeruli) onto thousands of Kenyon cells (KCs) through sparse
connections, then an inhibitory feedback (APL) keeps only the most active few percent. Dasgupta, Stevens & Navlakha
(Science 2017) showed that this "sparse expansion + winner-take-all" is a good similarity-search hash (FlyHash) and
beat the standard locality-sensitive hash (LSH), using RANDOM fly-like wiring. Here the projection is the real one,
read from FlyWire v783.

Pre-registered design (fixed before any result was seen):
  wiring   : excitatory uniglomerular PN -> KC synapses, pooled per glomerulus: 72 input channels x 4,797 KCs (KCs with
             no uniglomerular PN input are left out). Primary = binary connectivity (as in FlyHash); secondary =
             synapse-count weights.
  data     : MNIST (primary) and Fashion-MNIST, PCA-reduced to 72 dimensions (fit on the database only), one dimension
             per glomerulus. Which dimension goes to which glomerulus is arbitrary -> RUNS random assignments.
  hash     : centre each input (subtract its own mean), project, keep the top k KCs (winner-take-all) -> k-sparse
             binary code; k = 16, 32, 64 and 240 (~5% of KCs, the biological level).
  controls : RANDOM  -- every KC keeps its number of glomerulus inputs, glomeruli chosen uniformly (Dasgupta-style);
             SHUFFLE -- degree-preserving: every KC keeps its in-degree AND every glomerulus keeps how many KCs it
                        feeds (the real map's biases, e.g. toward food odours), partners randomised;
             LSH     -- the standard method: k random Gaussian directions, sign bits (Charikar 2002);
             each control is redrawn in every run.
  metric   : (primary) mean average precision of retrieving each query's true 2% nearest neighbours (Euclidean,
             in the 72-d input), 1,000 queries against a 10,000-image database, k = 32;
             (secondary) digit classification accuracy of a linear classifier on the codes vs on the raw 72-d input.
  criteria : the fly wiring "works as an AI layer" if it beats LSH; the real wiring is "better than random" only if
             it beats RANDOM and SHUFFLE in >= 95% of runs.
  EXPLORATORY (added after a 2-run smoke test, reported separately from the pre-registered result): the real map is
             very uneven (one glomerulus feeds >1,000 KCs, some feed none), because the fly gives more KCs to the
             odours that matter most. '*_sorted' variants give the highest-variance input dimensions to the most-connected
             glomeruli (importance -> more cells), for the real map and for SHUFFLE alike (RANDOM is unaffected).

Usage: python ml/flyhash.py --runs 20
"""
import argparse, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=20); ap.add_argument("--ks", default="16,32,64,240")
ap.add_argument("--db", type=int, default=10_000); ap.add_argument("--queries", type=int, default=1_000)
ap.add_argument("--datasets", default="mnist_784,Fashion-MNIST"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--clf_train", type=int, default=20_000); ap.add_argument("--clf_test", type=int, default=10_000)
args = ap.parse_args()
KS = [int(k) for k in args.ks.split(",")]; PRIMARY_K = 32


def real_wiring():
    """(glomeruli x KCs) excitatory synapse counts from uniglomerular PNs, and the glomerulus names"""
    from brain import Brain
    b = Brain("female"); t = np.array([str(x) for x in b.types["type"]]); cls = np.array([str(x) for x in b.types["class"]])
    kc = np.flatnonzero(cls == "Kenyon_Cell"); alpn = np.flatnonzero(cls == "ALPN")
    uni = [i for i in alpn if not t[i].startswith("M_") and "_" in t[i] and "PN" in t[i].split("_")[-1]]
    glom = sorted({t[i].split("_")[0] for i in uni}); gi = {g: k for k, g in enumerate(glom)}
    W = b.W.tocsr(); G = np.zeros((len(glom), kc.size), np.float32)
    for i in uni: G[gi[t[i].split("_")[0]]] += np.clip(W[i][:, kc].toarray().ravel(), 0, None)     # excitatory only
    keep = G.sum(0) > 0
    return G[:, keep], glom


def random_like(G, rng):
    """RANDOM: each KC keeps its in-degree; its glomeruli are chosen uniformly"""
    R = np.zeros_like(G); d = (G > 0).sum(0)
    for j in range(G.shape[1]): R[rng.choice(G.shape[0], d[j], replace=False), j] = 1.0
    return R


def shuffle_like(G, rng, swaps_per_edge=10):
    """SHUFFLE: degree-preserving rewiring of the bipartite graph (edge swaps that keep every glomerulus's out-degree
    and every KC's in-degree; swaps that would create a duplicate edge are skipped)"""
    A = (G > 0).copy(); gs, ks = np.nonzero(A); E = gs.size
    for _ in range(swaps_per_edge * E // 1000):                       # vectorised batches of 1000 proposed swaps
        a, b = rng.integers(0, E, 1000), rng.integers(0, E, 1000)
        for x, y in zip(a, b):
            g1, k1, g2, k2 = gs[x], ks[x], gs[y], ks[y]
            if g1 == g2 or k1 == k2 or A[g1, k2] or A[g2, k1]: continue
            A[g1, k1] = A[g2, k2] = False; A[g1, k2] = A[g2, k1] = True; ks[x], ks[y] = k2, k1
    return A.astype(np.float32)


def fly_codes(X, M, k):
    """centre each input, project through M (inputs x KCs), keep the top k -> boolean (n x KCs), exactly k per row"""
    Y = (X - X.mean(1, keepdims=True)) @ M
    top = np.argpartition(-Y, k - 1, axis=1)[:, :k]
    C = np.zeros(Y.shape, bool); np.put_along_axis(C, top, True, axis=1)
    return C


def lsh_codes(X, k, rng):
    return (X - X.mean(1, keepdims=True)) @ rng.standard_normal((X.shape[1], k)) > 0


def mean_ap(Q, D, truth, sim, rng):
    """mean average precision: rank the database by similarity to each query (ties broken at random, the same way for
    every method), and score how early the true nearest neighbours appear"""
    S = sim(Q, D) + 1e-3 * rng.random((Q.shape[0], D.shape[0]), dtype=np.float32)   # similarities are integers
    order = np.argsort(-S, axis=1)
    rel = np.zeros(S.shape, bool); np.put_along_axis(rel, truth, True, axis=1)
    rel = np.take_along_axis(rel, order, axis=1)                      # relevance in ranked order
    hits = np.cumsum(rel, axis=1); ranks = np.arange(1, S.shape[1] + 1)[None, :]
    return float(((hits / ranks) * rel).sum(1).mean() / truth.shape[1])


def overlap(Q, D): return Q.astype(np.float32) @ D.astype(np.float32).T                 # shared active cells
def hamming_sim(Q, D): return (2 * Q.astype(np.float32) - 1) @ (2 * D.astype(np.float32) - 1).T   # = bits - 2*Hamming


def load(name):
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml(name, version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    return X.astype(np.float32) / 255.0, y.astype(int)


def main():
    t_all = time.time(); G, glom = real_wiring(); B = (G > 0).astype(np.float32)
    n_in, n_kc = G.shape
    print(f"glomeruli that feed no Kenyon cell: {int(((G > 0).sum(1) == 0).sum())} of {n_in}")
    print(f"real wiring: {n_in} glomeruli x {n_kc:,} Kenyon cells, {int((G > 0).sum()):,} connections, "
          f"median {np.median((G > 0).sum(0)):.0f} glomeruli per KC", flush=True)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    out = dict(design=__doc__, n_glomeruli=n_in, n_kc=n_kc, results={})
    for ds in args.datasets.split(","):
        rng = np.random.default_rng(args.seed); Xall, yall = load(ds)
        idx = rng.permutation(len(Xall)); db, qu = idx[:args.db], idx[args.db:args.db + args.queries]
        pca = PCA(n_in, random_state=0).fit(Xall[db]); Z = pca.transform(Xall).astype(np.float32)
        Zd, Zq = Z[db], Z[qu]
        d2 = (Zq ** 2).sum(1)[:, None] + (Zd ** 2).sum(1)[None, :] - 2 * Zq @ Zd.T      # true neighbours, 72-d input
        truth = np.argsort(d2, axis=1)[:, :int(0.02 * args.db)]
        ctr = idx[args.db + args.queries:args.db + args.queries + args.clf_train]
        cte = idx[args.db + args.queries + args.clf_train:args.db + args.queries + args.clf_train + args.clf_test]
        R = {m: {k: [] for k in KS} for m in ("real", "real_weighted", "random", "shuffle", "lsh", "real_sorted", "shuffle_sorted")}
        pop = np.argsort(-(G > 0).sum(1), kind="stable")                 # glomeruli, most-connected first
        print(f"\n[{ds}] database {len(db):,}, queries {len(qu):,}; true neighbours = nearest 2%", flush=True)
        for r in range(args.runs):
            rr = np.random.default_rng(1000 + r); perm = rr.permutation(n_in)          # which PCA dim -> which glomerulus
            Sh = shuffle_like(G, rr)
            mats = {"real": B[perm], "real_weighted": G[perm], "random": random_like(G, rr)[perm], "shuffle": Sh[perm],
                    "real_sorted": B[pop], "shuffle_sorted": Sh[pop]}          # PCA dim i -> i-th most connected glomerulus
            for k in KS:
                for m, M in mats.items():
                    R[m][k].append(mean_ap(fly_codes(Zq, M, k), fly_codes(Zd, M, k), truth, overlap, rr))
                R["lsh"][k].append(mean_ap(lsh_codes(Zq, k, np.random.default_rng(5000 + r)), lsh_codes(Zd, k, np.random.default_rng(5000 + r)), truth, hamming_sim, rr))
            print(f"   run {r + 1:2d}/{args.runs}: mAP k={PRIMARY_K}  real {R['real'][PRIMARY_K][-1]:.3f}  random {R['random'][PRIMARY_K][-1]:.3f}  "
                  f"shuffle {R['shuffle'][PRIMARY_K][-1]:.3f}  LSH {R['lsh'][PRIMARY_K][-1]:.3f}", flush=True)
        # the "small AI": a linear classifier on the codes (first run's assignment) vs on the raw 72-d input
        rr = np.random.default_rng(1000); perm = rr.permutation(n_in); clf = {}
        def acc(Ftr, Fte):
            m = LogisticRegression(max_iter=300, C=1.0).fit(Ftr, yall[ctr]); return float((m.predict(Fte) == yall[cte]).mean())
        clf["raw 72-d input (no fly layer)"] = acc(Z[ctr], Z[cte])
        clf["real fly layer, k=240, sorted (exploratory)"] = acc(fly_codes(Z[ctr], B[pop], 240).astype(np.float32), fly_codes(Z[cte], B[pop], 240).astype(np.float32))
        for k in (PRIMARY_K, 240):
            for m, M in (("real", B[perm]), ("random", random_like(G, rr)[perm]), ("shuffle", shuffle_like(G, rr)[perm])):
                clf[f"{m} fly layer, k={k}"] = acc(fly_codes(Z[ctr], M, k).astype(np.float32), fly_codes(Z[cte], M, k).astype(np.float32))
        res = {m: {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), runs=v) for k, v in d.items()} for m, d in R.items()}
        win = {k: dict(real_beats_lsh=float(np.mean(np.array(R["real"][k]) > np.array(R["lsh"][k]))),
                       real_beats_random=float(np.mean(np.array(R["real"][k]) > np.array(R["random"][k]))),
                       real_beats_shuffle=float(np.mean(np.array(R["real"][k]) > np.array(R["shuffle"][k])))) for k in KS}
        out["results"][ds] = dict(mAP=res, wins=win, classification=clf)
        print(f"\n[{ds}] mean average precision of finding the true nearest neighbours (mean +- sd over {args.runs} runs)")
        print(f"{'k active':>9s} " + "".join(f"{m:>18s}" for m in R) + "   real beats: LSH / random / shuffle (share of runs)")
        for k in KS:
            print(f"{k:9d} " + "".join(f"{res[m][k]['mean']:11.3f}+-{res[m][k]['sd']:.3f}" for m in R)
                  + f"   {win[k]['real_beats_lsh']:.2f} / {win[k]['real_beats_random']:.2f} / {win[k]['real_beats_shuffle']:.2f}")
        print(f"\n[{ds}] digit/item classification accuracy (linear classifier, {len(ctr):,} train / {len(cte):,} test)")
        for n, a in clf.items(): print(f"   {n:34s} {a:.3f}")
        json.dump(out, open(os.path.join(HERE, "..", "atlas", "female", "ml", "flyhash.json"), "w"), indent=1)
    print(f"\ntotal {time.time() - t_all:.0f}s; saved atlas/female/ml/flyhash.json")


if __name__ == "__main__":
    main()
