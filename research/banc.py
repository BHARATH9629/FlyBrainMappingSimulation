"""BANC (a second female fly, brain + nerve cord; Harvard Dataverse doi:10.7910/DVN/7WTH1N, CC-BY 4.0) loaded into
the SAME simulator as FlyWire, without editing neuroredteam/brain.py.

Files (downloaded to the Desktop, see research/README section "Downloads"):
  C:/Users/hp/Desktop/ConnectomeLab_Downloads/BANC/banc_888_meta.feather            one row per neuron
  C:/Users/hp/Desktop/ConnectomeLab_Downloads/BANC/banc_888_edgelist_simple_v2.feather  neuron -> neuron synapse counts

Rules, identical to the FlyWire and MaleCNS loaders (Shiu et al. 2024): weight = synapse count; neurons whose
predicted transmitter is GABA or glutamate are inhibitory, all others excitatory. Glia are left out.
Cell types: the FlyWire-matched type (fafb_cell_type) where BANC has one, so behaviours can be defined by the same
names as in FlyWire; otherwise BANC's own cell_type.

scale: every weight is multiplied by this factor. The synapse-strength constant of the model was tuned on FlyWire's
synapse counts, and BANC's synapses come from a different detector, so research/PREREGISTRATION.md fixes ONE global
factor computed from the wiring only (matched cell types' input synapses), before any BANC behaviour is run.
"""
import os
import numpy as np, pandas as pd, scipy.sparse as sp
import guard  # noqa: F401  (puts neuroredteam/ on the path)
from brain import Brain

BANC_DIR = r"C:/Users/hp/Desktop/ConnectomeLab_Downloads/BANC"


class BancBrain(Brain):
    def __init__(self, scale=1.0):
        # deliberately NOT calling Brain.__init__ (it only knows FlyWire and MaleCNS); set the same attributes it sets
        self.sex = "female"; self.dataset = "BANC v888"
        meta = pd.read_feather(os.path.join(BANC_DIR, "banc_888_meta.feather"))
        meta = meta[meta["super_class"].astype(str) != "glia"].drop_duplicates("banc_888_id")
        self.ids = meta["banc_888_id"].astype("int64").values
        self.id2idx = {int(f): i for i, f in enumerate(self.ids)}; self.N = len(self.ids)
        ft = meta["fafb_cell_type"].astype(object); ct = meta["cell_type"].astype(object)
        typ = [a if isinstance(a, str) and a not in ("", "nan") else b for a, b in zip(ft, ct)]
        t = pd.DataFrame(index=np.arange(self.N))
        t["type"] = typ; t["flywireType"] = ft.values; t["bancType"] = ct.values
        t["class"] = meta["cell_class"].values; t["subclass"] = meta["cell_sub_class"].values
        t["superclass"] = meta["super_class"].values
        t["side"] = meta["side"].map({"left": "L", "right": "R", "center": "C"}).values
        t["nt"] = meta["neurotransmitter_predicted"].values
        self.types = t
        sign = np.where(pd.Series(meta["neurotransmitter_predicted"].astype(str)).str.contains("gaba|glutamate").values, -1.0, 1.0).astype(np.float32)
        e = pd.read_feather(os.path.join(BANC_DIR, "banc_888_edgelist_simple_v2.feather"), columns=["pre", "post", "count"])
        loc = pd.Series(np.arange(self.N), index=self.ids)
        pre = loc.reindex(e["pre"].astype("int64").values).values; post = loc.reindex(e["post"].astype("int64").values).values
        ok = ~(np.isnan(pre) | np.isnan(post)); pre, post = pre[ok].astype(np.int64), post[ok].astype(np.int64)
        w = e["count"].values[ok].astype(np.float32) * sign[pre] * np.float32(scale)
        self.W = sp.csr_matrix((w, (pre, post)), shape=(self.N, self.N))
        self.scale = scale; self._sub_cache = {}; self._run_cache = {}

    def positions(self):
        raise NotImplementedError("positions are not needed for the BANC analyses")


def input_synapses_by_type(brain, types):
    """median total input synapses per neuron, for each cell type in `types` (wiring only)"""
    tot = np.asarray(abs(brain.W).sum(0)).ravel(); T = np.array([str(x) for x in brain.types["type"]]); out = {}
    for ty in types:
        idx = np.flatnonzero(T == ty)
        if idx.size: out[ty] = float(np.median(tot[idx]))
    return out
