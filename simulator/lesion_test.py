"""
Experiment 2: trace the circuit from sugar neurons to the feeding motor neuron MN9,
then 'lesion' (silence) the key interneurons one at a time and see whether MN9 still fires.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import sugar_test as st          # loads the connectome (module-level) and exposes simulate()

ids, id2idx, W, N = st.ids, st.id2idx, st.W, st.N
mn9 = id2idx[st.MN9]
sugar = [id2idx[i] for i in st.SUGAR if i in id2idx]

# ---------- 1. graph analysis: how many synapses from tongue to muscle? ----------
# keep only strong excitatory connections (>= 5 synapses) so the path is a real pathway
Wexc = W.multiply(W >= 5 * st.W_SYN).tocsr()
adj = (Wexc > 0).astype(np.int8)
dist, pred, _src = dijkstra(adj, directed=True, indices=sugar, unweighted=True,
                      return_predecessors=True, min_only=True)
print(f"Shortest strong-excitatory path from sugar neurons to MN9: {int(dist[mn9])} synapses")
path = [mn9]
while pred[path[-1]] >= 0:
    path.append(pred[path[-1]])
path = path[::-1]
print("  path:", " -> ".join(("SUGAR " if p in sugar else "") + str(ids[p]) for p in path))
interneurons = [p for p in path if p not in sugar and p != mn9]

# how many synapses does each hop carry?
for a, b in zip(path[:-1], path[1:]):
    print(f"    {ids[a]} -> {ids[b]} : {int(W[a, b] / st.W_SYN)} synapses")

# ---------- 2. lesion experiments ----------
def simulate_with_lesion(stim_ids, silence_idx, seed=1):
    """Same as st.simulate but the silenced neurons never transmit spikes."""
    silence_idx = np.asarray(silence_idx)
    orig_W = st.W
    Wl = orig_W.tolil(copy=True)
    Wl[silence_idx, :] = 0            # zero all outgoing synapses (paper's definition of silencing)
    st.W = Wl.tocsr()
    try:
        return st.simulate(stim_ids, seed=seed)
    finally:
        st.W = orig_W

results = {}
print("\nBaseline (no lesion):")
c = st.simulate(st.SUGAR, seed=1); results["intact"] = c[mn9]
print(f"  MN9 = {c[mn9]} Hz")

for n in interneurons:
    t0 = time.time()
    c = simulate_with_lesion(st.SUGAR, [n])
    results[f"silence {ids[n]}"] = c[mn9]
    print(f"Silence interneuron {ids[n]}:  MN9 = {c[mn9]} Hz   ({time.time()-t0:.0f}s)")

# control lesion: silence a random neuron NOT on the path
rng = np.random.default_rng(7)
rand = int(rng.choice([i for i in np.flatnonzero(st.simulate(st.SUGAR, seed=1) == 0)][:5000]))
c = simulate_with_lesion(st.SUGAR, [rand])
results[f"silence random {ids[rand]}"] = c[mn9]
print(f"Silence random off-path neuron:  MN9 = {c[mn9]} Hz")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
names = list(results); vals = [results[k] for k in names]
ax.barh(names, vals, color=["#3ddc97"] + ["#ff5c6c"] * len(interneurons) + ["#8b98ad"])
ax.set_xlabel("MN9 firing rate (Hz) during sugar stimulation"); ax.invert_yaxis()
ax.set_title("Cutting the tongue-to-muscle circuit")
plt.tight_layout(); plt.savefig("lesion_test.png", dpi=130); print("saved lesion_test.png")
