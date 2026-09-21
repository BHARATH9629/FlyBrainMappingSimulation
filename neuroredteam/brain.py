"""
Fast whole-brain simulator for NeuroRedTeam.

Same leaky integrate-and-fire model and constants as Shiu et al. (Nature 2024) / simulator/sugar_test.py, but:
  * runs on the sub-network reachable from the stimulus (a few tens of thousands of neurons instead of 139k),
  * short trials (300 ms), and
  * supports interventions: lesion (silence), weaken (scale outgoing synapses), inject (extra Poisson input),
    bias (hold resting potential closer to threshold), potentiate (scale synapses onto a target neuron).
Results are cached so an agent that repeats an experiment pays nothing.
"""
import os, hashlib, json
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

V0, V_RST, V_TH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, R_POI, F_POI, DT = 0.275, 150.0, 250, 0.1


class Brain:
    """Two flies, kept strictly apart:
      sex="female": FlyWire v783 brain (Dorkenwald et al. 2024; model of Shiu et al. 2024), data/female_flywire/
      sex="male"  : MaleCNS v1.0 brain + nerve cord (Janelia/Google, Sep 2026),               data/male_malecns/
    Same model and constants for both; only the wiring table and the neuron naming differ."""
    def __init__(self, sex="female"):
        assert sex in ("female", "male"), sex
        self.sex = sex
        self.types = None                  # per-neuron cell type / class / side (pandas DataFrame, both sexes)
        if sex == "female":
            d = os.path.join(DATA, "female_flywire")
            comp = pd.read_csv(os.path.join(d, "Completeness_783.csv"), index_col=0)
            con = pd.read_parquet(os.path.join(d, "Connectivity_783.parquet"),
                                  columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
            self.ids = comp.index.values
            pre, post, w = con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values, con["Excitatory x Connectivity"].values
        else:
            pre, post, w = self._load_male()
        self.id2idx = {int(f): i for i, f in enumerate(self.ids)}
        self.N = len(self.ids)
        if sex == "female": self.types = self._load_female_types()
        self.W = sp.csr_matrix((w.astype(np.float32), (pre, post)), shape=(self.N, self.N))  # signed synapse COUNTS, W[pre, post]
        self._sub_cache = {}
        self._run_cache = {}

    def _load_female_types(self):
        """Cell types for the female (Schlegel et al. 2024 annotations, same columns as the male table so that
        by_type() works for both sexes). Missing file -> None (only needed for type-named behaviours)."""
        f = os.path.join(DATA, "female_flywire", "flywire_annotations.tsv")
        if not os.path.exists(f): return None
        a = pd.read_csv(f, sep="\t", usecols=["root_id", "cell_type", "cell_class", "cell_sub_class", "super_class", "side"],
                        low_memory=False)
        a = a.drop_duplicates("root_id").set_index("root_id").reindex(self.ids)
        t = pd.DataFrame(index=np.arange(self.N))
        t["type"] = a["cell_type"].values; t["flywireType"] = a["cell_type"].values
        t["class"] = a["cell_class"].values; t["subclass"] = a["cell_sub_class"].values; t["superclass"] = a["super_class"].values
        t["side"] = a["side"].map({"left": "L", "right": "R", "center": "C"}).values
        return t

    def _load_male(self):
        """MaleCNS flat connectome (CC-BY, male-cns.janelia.org). Sign rule as in Shiu et al. 2024: GABA and glutamate
        inhibitory, everything else excitatory. Weight = synapse count."""
        d = os.path.join(DATA, "male_malecns")
        ann = pd.read_feather(os.path.join(d, "body-annotations-male-cns-v1.0-minconf-0.5.feather"),
                              columns=["bodyId", "type", "flywireType", "superclass", "class", "subclass", "somaSide", "rootSide", "somaLocation", "receptorType", "fruDsx"])
        nt = pd.read_feather(os.path.join(d, "body-neurotransmitters-male-cns-v1.0.feather"), columns=["body", "consensus_nt"])
        ann = ann.drop_duplicates("bodyId").set_index("bodyId")
        ann["side"] = ann["somaSide"].fillna(ann["rootSide"])
        bodies = ann.index.values.astype(np.int64)          # 211,577 annotated neurons; fragments are dropped
        self.ids = bodies; self.types = ann
        loc = pd.Series(np.arange(len(bodies)), index=bodies)
        sign = nt.drop_duplicates("body").set_index("body")["consensus_nt"].map(
            lambda x: -1.0 if x in ("gaba", "glutamate") else 1.0).reindex(bodies).fillna(1.0).values.astype(np.float32)
        # the weights file is 152 M rows (all bodies); stream it and keep only neuron-to-neuron edges
        import pyarrow.feather as pf
        pres, posts, ws = [], [], []
        for batch in pf.read_table(os.path.join(d, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")).to_batches(2_000_000):
            bp = batch.column("body_pre").to_numpy(); bq = batch.column("body_post").to_numpy(); bw = batch.column("weight").to_numpy()
            ip = loc.reindex(bp).values; iq = loc.reindex(bq).values
            ok = ~(np.isnan(ip) | np.isnan(iq))
            pres.append(ip[ok].astype(np.int64)); posts.append(iq[ok].astype(np.int64)); ws.append(bw[ok].astype(np.float32))
        pre, post, w = np.concatenate(pres), np.concatenate(posts), np.concatenate(ws)
        w = w * sign[pre]
        return pre, post, w

    def positions(self):
        """(N, 3) soma / representative position per neuron in dataset voxels (nan if unknown). For drawing."""
        if getattr(self, "_pos", None) is not None: return self._pos
        P = np.full((self.N, 3), np.nan, np.float32)
        if self.sex == "female":
            a = pd.read_csv(os.path.join(DATA, "female_flywire", "flywire_annotations.tsv"), sep="	",
                            usecols=["root_id", "pos_x", "pos_y", "pos_z"], low_memory=False).drop_duplicates("root_id")
            a = a.set_index("root_id").reindex(self.ids)
            P[:] = a[["pos_x", "pos_y", "pos_z"]].values
        else:
            loc = self.types["somaLocation"] if "somaLocation" in self.types else None
            if loc is not None:
                ok = loc.notna().values
                P[ok] = np.stack(loc[ok].values)
        # neurons without a soma in the volume (sensory afferents): put them at the centroid of their targets
        miss = np.flatnonzero(~np.isfinite(P[:, 0]))
        if miss.size:
            A = (abs(self.W[miss]) > 0).astype(np.float32)
            okp = np.isfinite(P[:, 0]).astype(np.float32)
            cnt = A @ okp
            fill = np.stack([(A @ np.nan_to_num(P[:, k])) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
            good = cnt > 0
            P[miss[good]] = fill[good]
        self._pos = P
        return P

    def by_type(self, types=None, side=None, subclass=None, superclass=None, exclude_types=(), regex=None, receptor=None):
        """Indices of neurons by cell type / subclass / superclass (either sex), optionally one side 'L'/'R'."""
        if self.types is None: raise RuntimeError("no type table: run data/get_data.py to fetch the annotations")
        T = self.types
        m = np.ones(self.N, bool)
        if types is not None: m &= T["type"].isin(list(types)).values
        if regex is not None: m &= T["type"].astype(str).str.fullmatch(regex).values
        if subclass is not None: m &= T["subclass"].isin([subclass] if isinstance(subclass, str) else list(subclass)).values
        if superclass is not None: m &= T["superclass"].isin([superclass] if isinstance(superclass, str) else list(superclass)).values
        if exclude_types: m &= ~T["type"].isin(list(exclude_types)).values
        if receptor is not None and "receptorType" in T: m &= T["receptorType"].astype(str).str.contains(receptor).values
        if side: m &= (T["side"] == side).values
        return np.flatnonzero(m)

    def type_of(self, idx):
        """Cell type name of a neuron index ('' if unknown)."""
        if self.types is None: return ""
        v = self.types["type"].iloc[int(idx)]
        return "" if pd.isna(v) else str(v)

    # ---------------------------------------------------------------- graph helpers
    def idx(self, flywire_ids):
        return np.array([self.id2idx[int(i)] for i in flywire_ids if int(i) in self.id2idx], dtype=int)

    def inputs_to(self, target, top=30, excitatory_only=False):
        """Presynaptic partners of `target` (index), strongest first. Returns list of (index, signed synapses)."""
        col = self.W[:, target].toarray().ravel()
        order = np.argsort(-np.abs(col))
        out = [(int(i), int(col[i])) for i in order if col[i] != 0 and (col[i] > 0 or not excitatory_only)]
        return out[:top]

    def outputs_of(self, source, top=30):
        row = self.W[source].toarray().ravel(); order = np.argsort(-np.abs(row))
        return [(int(j), int(row[j])) for j in order if row[j] != 0][:top]

    def subnetwork(self, seeds, depth=3, min_syn=5, must_include=()):
        """Neurons reachable from `seeds` within `depth` hops over connections with >= min_syn synapses."""
        key = (tuple(sorted(map(int, seeds))), depth, min_syn, tuple(sorted(map(int, must_include))))
        if key in self._sub_cache: return self._sub_cache[key]
        A = (abs(self.W) >= min_syn).astype(np.int8).tocsr()
        reached = np.zeros(self.N, bool); reached[list(seeds)] = True
        frontier = np.array(list(seeds), dtype=int)
        for _ in range(depth):
            nxt = np.unique(A[frontier].indices)
            nxt = nxt[~reached[nxt]]
            if nxt.size == 0: break
            reached[nxt] = True; frontier = nxt
        reached[list(must_include)] = True
        sub = np.flatnonzero(reached)
        self._sub_cache[key] = sub
        return sub

    # ---------------------------------------------------------------- simulation
    def run(self, stimulate, t_ms=300, lesion=(), weaken=None, inject=None, bias=None, potentiate=None,
            record=(), seed=1, depth=3, min_syn=5, spikes=False):
        """
        stimulate : indices driven at 150 Hz (the sensory input)
        lesion    : indices whose outgoing synapses are removed (silenced)
        weaken    : {index: factor in (0,1]} scale outgoing synapses
        inject    : {index: rate_hz} extra Poisson drive
        bias      : {index: mV} resting potential raised (capped below threshold)
        potentiate: {(pre, post): factor} scale a specific synapse
        record    : indices whose firing rate (Hz) is returned
        spikes    : also return every spike as (time_ms, global index) arrays (for animation; not cached)
        """
        stimulate = tuple(sorted(map(int, stimulate))); record = tuple(sorted(map(int, record)))
        lesion = tuple(sorted(map(int, lesion)))
        weaken = dict(sorted((int(k), float(v)) for k, v in (weaken or {}).items()))
        inject = dict(sorted((int(k), float(v)) for k, v in (inject or {}).items()))
        bias = dict(sorted((int(k), float(v)) for k, v in (bias or {}).items()))
        potentiate = dict(sorted(((int(a), int(b)), float(v)) for (a, b), v in (potentiate or {}).items()))
        key = hashlib.md5(json.dumps([stimulate, t_ms, lesion, weaken, list(inject.items()), list(bias.items()),
                                      [(list(k), v) for k, v in potentiate.items()], record, seed, depth, min_syn]).encode()).hexdigest()
        if key in self._run_cache and not spikes: return self._run_cache[key]

        seeds = list(stimulate) + list(inject.keys())
        must = list(record) + list(lesion) + list(weaken) + list(bias) + [p for pr in potentiate for p in pr]
        sub = self.subnetwork(seeds, depth, min_syn, must_include=must)
        n = sub.size; loc = {int(g): i for i, g in enumerate(sub)}
        # base sub-matrix is cached per sub-network; interventions are applied as cheap sparse ops
        skey = ("W", tuple(sub.tolist()))
        if skey not in self._sub_cache:
            self._sub_cache[skey] = (self.W[sub][:, sub].astype(np.float32) * W_SYN).tocsr()
        Wl = self._sub_cache[skey]
        scale = np.ones(n, np.float32)
        for g in lesion:
            if g in loc: scale[loc[g]] = 0.0
        for g, f in weaken.items():
            if g in loc: scale[loc[g]] *= f
        if (scale != 1).any():
            Wl = sp.diags(scale) @ Wl                     # scale outgoing synapses row-wise
        if potentiate:
            rows, cols, vals = [], [], []
            for (a, b), f in potentiate.items():
                if a in loc and b in loc:
                    w = Wl[loc[a], loc[b]]
                    if w != 0: rows.append(loc[a]); cols.append(loc[b]); vals.append((f - 1.0) * w)
            if rows:
                Wl = (Wl + sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)).tocsr()
        Wl = Wl.tocsr()

        rng = np.random.default_rng(seed)
        v0 = np.full(n, V0, np.float32)
        for g, mv in bias.items():
            if g in loc: v0[loc[g]] = min(V_TH - 0.5, V0 + mv)
        stim = np.array([loc[g] for g in stimulate if g in loc], dtype=int)
        inj_idx = np.array([loc[g] for g in inject if g in loc], dtype=int)
        inj_p = np.array([inject[g] * DT / 1000 for g in inject if g in loc], dtype=np.float32)
        v = v0.copy(); gsyn = np.zeros(n, np.float32); refr = np.zeros(n, np.float32)
        rfc = np.full(n, T_RFC, np.float32); rfc[stim] = 0
        n_steps = int(t_ms / DT); n_delay = int(round(T_DLY / DT))
        queue = [np.empty(0, np.int64) for _ in range(n_delay)]
        counts = np.zeros(n, np.int64); spk_t, spk_i = [], []
        p_poi = R_POI * DT / 1000; w_poi = W_SYN * F_POI; dv = DT / T_MBR; dg = np.float32(np.exp(-DT / TAU))
        for _ in range(n_steps):   # _ is the step index (used for spike times)
            arriving = queue.pop(0)
            if arriving.size: gsyn += np.asarray(Wl[arriving].sum(axis=0)).ravel()
            if stim.size:
                hit = stim[rng.random(stim.size) < p_poi]; v[hit] += w_poi
            if inj_idx.size:
                hit = inj_idx[rng.random(inj_idx.size) < inj_p]; v[hit] += w_poi
            act = refr <= 0
            v[act] += dv * (v0[act] - v[act] + gsyn[act]); gsyn[act] *= dg; refr -= DT
            spk = np.flatnonzero(v > V_TH)
            if spk.size:
                v[spk] = V_RST; gsyn[spk] = 0; refr[spk] = rfc[spk]; counts[spk] += 1
                if spikes: spk_t.append(np.full(spk.size, _ * DT, np.float32)); spk_i.append(sub[spk])
            queue.append(spk)
        hz = counts * (1000.0 / t_ms)
        out = {"rates": {int(g): float(hz[loc[g]]) if g in loc else 0.0 for g in record},
               "active": {int(sub[i]): float(hz[i]) for i in np.flatnonzero(counts)},
               "n_sub": int(n)}
        if spikes:
            out = dict(out, spike_t=np.concatenate(spk_t) if spk_t else np.empty(0, np.float32),
                       spike_i=np.concatenate(spk_i) if spk_i else np.empty(0, np.int64))
            return out
        self._run_cache[key] = out
        return out
