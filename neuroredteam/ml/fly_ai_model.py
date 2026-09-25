"""Build the Fly Brain Digit Reader model: a small AI whose hidden layer is the real fly mushroom-body wiring.

  image (28x28) -> PCA to 72 numbers -> one number per glomerulus (most informative -> most-connected glomerulus)
  -> the real FlyWire projection-neuron -> Kenyon-cell wiring (72 x 4,804, excitatory, binary; FIXED, never trained)
  -> only the 240 most driven Kenyon cells stay on (~5%, winner-take-all, as the APL neuron does in the fly)
  -> a linear reader (logistic regression) -> digit 0-9.  Only the reader and the PCA are fitted.

Saves ml/fly_ai_model.npz (everything the app needs, so it starts instantly) and prints the test accuracy on the
10,000 MNIST test images the model never saw.
Usage: python ml/fly_ai_model.py
"""
import os, sys, time, warnings
import numpy as np
import scipy.sparse as sp
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
K_ACTIVE = 240


def main():
    t0 = time.time()
    from brain import Brain
    b = Brain("female"); t = np.array([str(x) for x in b.types["type"]]); cls = np.array([str(x) for x in b.types["class"]])
    kc = np.flatnonzero(cls == "Kenyon_Cell"); alpn = np.flatnonzero(cls == "ALPN")
    uni = [i for i in alpn if not t[i].startswith("M_") and "_" in t[i] and "PN" in t[i].split("_")[-1]]
    glom = sorted({t[i].split("_")[0] for i in uni}); gi = {g: k for k, g in enumerate(glom)}
    W = b.W.tocsr(); G = np.zeros((len(glom), kc.size), np.float32)
    for i in uni: G[gi[t[i].split("_")[0]]] += np.clip(W[i][:, kc].toarray().ravel(), 0, None)
    keep = G.sum(0) > 0; G = G[:, keep]; kc = kc[keep]
    pop = np.argsort(-(G > 0).sum(1), kind="stable")                  # glomeruli, most-connected first
    M = (G[pop] > 0).astype(np.float32)                                # row i = the glomerulus that gets PCA dim i
    P = b.positions(); pos = P[kc]                                     # where each Kenyon cell sits in the brain
    okp = np.flatnonzero(np.isfinite(P).all(1))
    brain_pts = P[np.random.default_rng(1).choice(okp, 30000, replace=False)]   # a faint outline of the whole brain
    print(f"fly layer: {M.shape[0]} glomeruli -> {M.shape[1]:,} Kenyon cells (real FlyWire wiring)", flush=True)

    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X = X.astype(np.float32) / 255.0; y = y.astype(int)
    Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]    # the standard MNIST split
    pca = PCA(M.shape[0], random_state=0).fit(Xtr)

    def codes(Xs):
        Z = pca.transform(Xs).astype(np.float32); Y = (Z - Z.mean(1, keepdims=True)) @ M
        top = np.argpartition(-Y, K_ACTIVE - 1, axis=1)[:, :K_ACTIVE]
        rows = np.repeat(np.arange(len(Xs)), K_ACTIVE)
        return sp.csr_matrix((np.ones(rows.size, np.float32), (rows, top.ravel())), shape=(len(Xs), M.shape[1]))
    Ctr, Cte = codes(Xtr), codes(Xte)
    clf = LogisticRegression(max_iter=500, C=1.0).fit(Ctr, ytr)
    acc = float((clf.predict(Cte) == yte).mean())
    print(f"test accuracy on the 10,000 unseen MNIST test digits: {acc * 100:.1f}%  ({time.time() - t0:.0f}s)", flush=True)

    keep_db = np.random.default_rng(0).choice(60000, 6000, replace=False)   # small gallery for 'most similar digits'
    np.savez_compressed(os.path.join(HERE, "fly_ai_model.npz"), pca_mean=pca.mean_.astype(np.float32),
                        pca_comp=pca.components_.astype(np.float32), M=M, k=K_ACTIVE, coef=clf.coef_.astype(np.float32),
                        intercept=clf.intercept_.astype(np.float32), acc=acc, kc_ids=b.ids[kc].astype(np.int64),
                        kc_pos=pos.astype(np.float32), brain_pts=brain_pts.astype(np.float32), glom=np.array(glom)[pop], n_train=60000, n_test=10000,
                        gal_img=(Xtr[keep_db] * 255).astype(np.uint8), gal_lab=ytr[keep_db].astype(np.int8),
                        gal_codes=np.asarray(Ctr[keep_db].toarray(), bool),
                        test_img=(Xte[:500] * 255).astype(np.uint8), test_lab=yte[:500].astype(np.int8))   # unseen digits for the demo
    print("saved ml/fly_ai_model.npz")


if __name__ == "__main__":
    main()
