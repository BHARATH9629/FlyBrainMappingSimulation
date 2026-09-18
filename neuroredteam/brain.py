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
    def __init__(self):
        comp = pd.read_csv(os.path.join(DATA, "Completeness_783.csv"), index_col=0)
        con = pd.read_parquet(os.path.join(DATA, "Connectivity_783.parquet"),
                              columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        self.ids = comp.index.values
        self.id2idx = {int(f): i for i, f in enumerate(self.ids)}
        self.N = len(self.ids)
        self.W = sp.csr_matrix((con["Excitatory x Connectivity"].values.astype(np.float32),
                                (con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values)),
                               shape=(self.N, self.N))          # signed synapse COUNTS, W[pre, post]
        self._sub_cache = {}
        self._run_cache = {}

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
            record=(), seed=1, depth=3, min_syn=5):
        """
        stimulate : indices driven at 150 Hz (the sensory input)
        lesion    : indices whose outgoing synapses are removed (silenced)
        weaken    : {index: factor in (0,1]} scale outgoing synapses
        inject    : {index: rate_hz} extra Poisson drive
        bias      : {index: mV} resting potential raised (capped below threshold)
        potentiate: {(pre, post): factor} scale a specific synapse
        record    : indices whose firing rate (Hz) is returned
        """
        stimulate = tuple(sorted(map(int, stimulate))); record = tuple(sorted(map(int, record)))
        lesion = tuple(sorted(map(int, lesion)))
        weaken = dict(sorted((int(k), float(v)) for k, v in (weaken or {}).items()))
        inject = dict(sorted((int(k), float(v)) for k, v in (inject or {}).items()))
        bias = dict(sorted((int(k), float(v)) for k, v in (bias or {}).items()))
        potentiate = dict(sorted(((int(a), int(b)), float(v)) for (a, b), v in (potentiate or {}).items()))
        key = hashlib.md5(json.dumps([stimulate, t_ms, lesion, weaken, list(inject.items()), list(bias.items()),
                                      [(list(k), v) for k, v in potentiate.items()], record, seed, depth, min_syn]).encode()).hexdigest()
        if key in self._run_cache: return self._run_cache[key]

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
        counts = np.zeros(n, np.int64)
        p_poi = R_POI * DT / 1000; w_poi = W_SYN * F_POI; dv = DT / T_MBR; dg = np.float32(np.exp(-DT / TAU))
        for _ in range(n_steps):
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
            queue.append(spk)
        hz = counts * (1000.0 / t_ms)
        out = {"rates": {int(g): float(hz[loc[g]]) if g in loc else 0.0 for g in record},
               "active": {int(sub[i]): float(hz[i]) for i in np.flatnonzero(counts)},
               "n_sub": int(n)}
        self._run_cache[key] = out
        return out
