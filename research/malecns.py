"""MaleCNS v1.0 (a male fly, brain + nerve cord; CC-BY, male-cns.janelia.org) loaded into the SAME simulator, keeping
ONLY real traced neurons -- without editing neuroredteam/brain.py.

Why a new loader: the project's original male loader (brain.py) keeps every annotated body, 211,577 of them, which
includes 15,925 orphan fragments, 11,864 glia, 10,751 'unimportant' bodies and 1,832 still being assigned. Here only
bodies with status "Traced" or "Anchor" are kept (165,733; the MaleCNS / Codex count of neurons is ~166,700).
Rules otherwise identical to the other loaders: weight = synapse count; GABA and glutamate inhibitory; cell type =
MaleCNS's own type where available (finer), otherwise the FlyWire-matched type; side = soma side, else root side.

The filtered weight matrix is cached in research/results/male/ (gitignored .npz) because the raw weights file has
~152 million rows and takes minutes to stream.
"""
import os
import numpy as np, pandas as pd, scipy.sparse as sp
import guard  # noqa: F401
from brain import Brain

D = os.path.join(guard.ROOT, "data", "male_malecns")
CACHE = os.path.join(guard.ROOT, "research", "results", "male", "malecns_traced_W.npz")
KEEP_STATUS = ("Traced", "Anchor")


class MaleBrain(Brain):
    def __init__(self, scale=1.0):
        self.sex = "male"; self.dataset = "MaleCNS v1.0 (traced neurons only)"
        a = pd.read_feather(os.path.join(D, "body-annotations-male-cns-v1.0-minconf-0.5.feather"))
        a = a[a["status"].astype(str).isin(KEEP_STATUS)].drop_duplicates("bodyId").set_index("bodyId")
        self.ids = a.index.values.astype(np.int64); self.N = len(self.ids)
        self.id2idx = {int(f): i for i, f in enumerate(self.ids)}
        ft = a["flywireType"].astype(object); ty = a["type"].astype(object)
        t = pd.DataFrame(index=np.arange(self.N))
        # the male's OWN type first (finer: it separates sugar / water taste neurons and names MN9), else FlyWire's
        t["type"] = [g if isinstance(g, str) and g not in ("", "nan") else f for f, g in zip(ft, ty)]
        t["flywireType"] = ft.values; t["maleType"] = ty.values
        t["class"] = a["class"].values; t["subclass"] = a["subclass"].values; t["superclass"] = a["superclass"].values
        t["side"] = a["somaSide"].fillna(a["rootSide"]).values; t["receptorType"] = a["receptorType"].values
        self.types = t
        if os.path.exists(CACHE):
            W = sp.load_npz(CACHE).tocsr()
        else:
            nt = pd.read_feather(os.path.join(D, "body-neurotransmitters-male-cns-v1.0.feather"), columns=["body", "consensus_nt"])
            sign = nt.drop_duplicates("body").set_index("body")["consensus_nt"].map(
                lambda x: -1.0 if x in ("gaba", "glutamate") else 1.0).reindex(self.ids).fillna(1.0).values.astype(np.float32)
            import pyarrow.feather as pf
            loc = pd.Series(np.arange(self.N), index=self.ids); P, Q, V = [], [], []
            for batch in pf.read_table(os.path.join(D, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")).to_batches(2_000_000):
                ip = loc.reindex(batch.column("body_pre").to_numpy()).values; iq = loc.reindex(batch.column("body_post").to_numpy()).values
                ok = ~(np.isnan(ip) | np.isnan(iq))
                P.append(ip[ok].astype(np.int64)); Q.append(iq[ok].astype(np.int64)); V.append(batch.column("weight").to_numpy()[ok].astype(np.float32))
            pre, post, w = np.concatenate(P), np.concatenate(Q), np.concatenate(V)
            W = sp.csr_matrix((w * sign[pre], (pre, post)), shape=(self.N, self.N))
            os.makedirs(os.path.dirname(CACHE), exist_ok=True); sp.save_npz(CACHE, W)
        self.W = (W * np.float32(scale)).tocsr() if scale != 1.0 else W
        self.scale = scale; self._sub_cache = {}; self._run_cache = {}

    def positions(self):
        raise NotImplementedError("positions are not needed for the male analyses")
