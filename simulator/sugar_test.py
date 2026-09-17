"""
Minimal whole-fly-brain simulation (FlyWire v783, 138,639 neurons, 15.1M synapses).

Reproduces the headline experiment of Shiu et al., Nature 2024:
  stimulate 21 sugar-sensing neurons on the tongue  ->  does the feeding
  motor neuron MN9 (which extends the proboscis) fire?

Leaky integrate-and-fire model, same equations/constants as the paper's model.py:
  dv/dt = (v0 - v + g) / t_mbr ,  dg/dt = -g / tau ,  spike when v > v_th
  on presynaptic spike (after 1.8 ms delay): g += w,  w = (signed synapse count) * 0.275 mV
"""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "")

# ---------------- constants (from model.py / Kakaria & de Bivort 2017) ----------------
V0, V_RST, V_TH = -52.0, -52.0, -45.0   # mV
T_MBR, TAU      = 20.0, 5.0             # ms
T_RFC, T_DLY    = 2.2, 1.8              # ms
W_SYN           = 0.275                 # mV per synapse
R_POI, F_POI    = 150.0, 250            # Hz, Poisson weight scaling
DT              = 0.1                   # ms
T_RUN           = 1000.0                # ms per trial

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265   # proboscis motor neuron (feeding)

# ---------------- load the connectome ----------------
t0 = time.time()
comp = pd.read_csv(DATA + "Completeness_783.csv", index_col=0)
ids = comp.index.values
id2idx = {fid: i for i, fid in enumerate(ids)}
N = len(ids)
con = pd.read_parquet(DATA + "Connectivity_783.parquet",
                      columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
W = sp.csr_matrix((con["Excitatory x Connectivity"].values.astype(np.float32) * W_SYN,
                   (con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values)),
                  shape=(N, N))          # W[pre, post] = weight in mV
del con
print(f"Loaded {N:,} neurons, {W.nnz:,} synapses in {time.time()-t0:.1f}s")


def simulate(stim_ids, seed=0):
    """Run one 1-second trial stimulating `stim_ids` with 150 Hz Poisson input.
    Returns dict {neuron index: number of spikes}."""
    rng = np.random.default_rng(seed)
    stim = np.array([id2idx[i] for i in stim_ids if i in id2idx])
    print(f"  stimulating {stim.size} of {len(stim_ids)} requested neurons (rest missing in v783)")
    v = np.full(N, V0, dtype=np.float32)
    g = np.zeros(N, dtype=np.float32)
    refr = np.zeros(N, dtype=np.float32)       # remaining refractory time (ms)
    rfc = np.full(N, T_RFC, dtype=np.float32)
    rfc[stim] = 0.0                            # stimulated neurons: no refractory period (as in paper)
    n_steps = int(T_RUN / DT)
    n_delay = int(round(T_DLY / DT))
    queue = [np.empty(0, dtype=np.int64) for _ in range(n_delay)]   # spikes in flight
    counts = np.zeros(N, dtype=np.int64)
    p_poi = R_POI * DT / 1000.0                # prob. of a Poisson event per step
    w_poi = W_SYN * F_POI
    dv = DT / T_MBR
    dg = np.float32(np.exp(-DT / TAU))

    for step in range(n_steps):
        # deliver spikes that were emitted T_DLY ago
        arriving = queue.pop(0)
        if arriving.size:
            g += np.asarray(W[arriving].sum(axis=0)).ravel()
        # Poisson drive straight into v of stimulated neurons
        hit = stim[rng.random(stim.size) < p_poi]
        v[hit] += w_poi
        # integrate (only non-refractory neurons update)
        active = refr <= 0
        v[active] += dv * (V0 - v[active] + g[active])
        g[active] *= dg
        refr -= DT
        # spikes
        spk = np.flatnonzero(v > V_TH)
        if spk.size:
            v[spk] = V_RST
            g[spk] = 0.0
            refr[spk] = rfc[spk]
            counts[spk] += 1
        queue.append(spk)
    return counts


if __name__ == "__main__":
    results = {}
    # Experiment 1: sugar on the tongue
    t0 = time.time()
    c_sugar = simulate(SUGAR, seed=1)
    print(f"\n[sugar]   simulated 1 s of brain time in {time.time()-t0:.1f}s")
    results["sugar"] = c_sugar
    # Experiment 2: control - 21 random neurons instead of sugar neurons
    rng = np.random.default_rng(42)
    rand_ids = rng.choice(ids, 21, replace=False)
    t0 = time.time()
    c_rand = simulate(rand_ids, seed=1)
    print(f"[control] simulated 1 s of brain time in {time.time()-t0:.1f}s")
    results["random"] = c_rand

    mn9 = id2idx[MN9]
    for name, c in results.items():
        active = int((c > 0).sum())
        print(f"\n=== {name} ===")
        print(f"  neurons that fired at all : {active:,} of {N:,}")
        print(f"  total spikes              : {c.sum():,}")
        print(f"  MN9 (feeding motor neuron): {c[mn9]} Hz")
        top = np.argsort(c)[::-1][:8]
        print("  top firing neurons (flywire id : Hz):")
        for i in top:
            tag = " <- sugar input" if ids[i] in SUGAR else (" <- MN9 !" if i == mn9 else "")
            print(f"    {ids[i]} : {c[i]}{tag}")

    # save raster-style summary plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, (name, c) in enumerate(results.items()):
        act = np.flatnonzero(c > 0)
        ax[0].scatter(np.full(act.size, k), c[act], s=6, alpha=.5, label=name)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(results.keys())
    ax[0].set_ylabel("firing rate (Hz)"); ax[0].set_title("Every neuron that fired")
    ax[1].bar(list(results.keys()), [results[k][mn9] for k in results], color=["#3ddc97", "#8b98ad"])
    ax[1].set_ylabel("MN9 rate (Hz)"); ax[1].set_title("Proboscis motor neuron MN9")
    plt.tight_layout(); plt.savefig("sugar_test.png", dpi=130)
    print("\nsaved sugar_test.png")
