"""Animated demo: watch the sugar signal ripple through the fly brain, hop by hop, until the feeding
motor neuron fires. Neurons are placed left-to-right by how many synapses they are from the tongue."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sugar_test as st

ids, id2idx, W, N = st.ids, st.id2idx, st.W, st.N
mn9 = id2idx[st.MN9]
sugar = np.array([id2idx[i] for i in st.SUGAR if i in id2idx])

# ---- run 400 ms and keep every spike with its time ----
T_MS = 400
st.T_RUN = float(T_MS)
spikes_t, spikes_i = [], []
_orig_append = list.append
def simulate_record(stim_ids, seed=1):
    rng = np.random.default_rng(seed)
    stim = np.array([id2idx[i] for i in stim_ids if i in id2idx])
    v = np.full(N, st.V0, np.float32); g = np.zeros(N, np.float32); refr = np.zeros(N, np.float32)
    rfc = np.full(N, st.T_RFC, np.float32); rfc[stim] = 0
    n_steps = int(st.T_RUN / st.DT); n_delay = int(round(st.T_DLY / st.DT))
    queue = [np.empty(0, np.int64) for _ in range(n_delay)]
    p_poi = st.R_POI * st.DT / 1000; w_poi = st.W_SYN * st.F_POI; dv = st.DT / st.T_MBR; dg = np.float32(np.exp(-st.DT / st.TAU))
    for step in range(n_steps):
        arriving = queue.pop(0)
        if arriving.size: g += np.asarray(W[arriving].sum(axis=0)).ravel()
        hit = stim[rng.random(stim.size) < p_poi]; v[hit] += w_poi
        act = refr <= 0
        v[act] += dv * (st.V0 - v[act] + g[act]); g[act] *= dg; refr -= st.DT
        spk = np.flatnonzero(v > st.V_TH)
        if spk.size:
            v[spk] = st.V_RST; g[spk] = 0; refr[spk] = rfc[spk]
            spikes_t.append(np.full(spk.size, step * st.DT)); spikes_i.append(spk)
        queue.append(spk)
simulate_record(st.SUGAR)
t_all = np.concatenate(spikes_t); i_all = np.concatenate(spikes_i)
active = np.unique(i_all)
print(f"{active.size} neurons fired in {T_MS} ms; MN9 spikes = {(i_all == mn9).sum()}")

# ---- layout: x = synaptic hops from the sugar neurons (strong excitatory links), y = spread ----
adj = (W.multiply(W >= 3 * st.W_SYN) > 0).astype(np.int8)
hops = dijkstra(adj, directed=True, indices=sugar, unweighted=True, min_only=True, limit=6)
hops = np.where(np.isfinite(hops), hops, 6)
rng = np.random.default_rng(0)
x = hops[active] + rng.uniform(-0.35, 0.35, active.size)
y = rng.uniform(0, 1, active.size)
x[active == mn9] = hops[mn9]; y[active == mn9] = 0.5
pos = {n: k for k, n in enumerate(active)}

# ---- animation ----
FRAME_MS, GLOW_MS = 4, 12
frames = int(T_MS / FRAME_MS)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [4, 1.2]}, facecolor="#0b0f17")
for a in (ax, ax2): a.set_facecolor("#0b0f17"); a.tick_params(colors="#8b98ad")
ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.05, 1.05); ax.set_yticks([])
ax.set_xticks(range(7)); ax.set_xticklabels(["tongue\n(sugar sensors)", "1 hop", "2 hops", "3 hops", "4 hops", "5 hops", "6+"], color="#e6edf7")
ax.set_title("Fly brain, live: sugar on the tongue → signal spreads synapse by synapse → feeding neuron MN9 fires",
             color="#e6edf7", fontsize=12)
base = ax.scatter(x, y, s=10, c="#233044", zorder=1)
glow = ax.scatter([], [], s=[], c=[], cmap="autumn", vmin=0, vmax=1, zorder=2)
mn9_dot = ax.scatter([hops[mn9]], [0.5], s=260, facecolors="none", edgecolors="#3ddc97", linewidths=2, zorder=3)
ax.annotate("MN9 (feeding\nmotor neuron)", (hops[mn9], 0.5), xytext=(hops[mn9] + 0.5, 0.85), color="#3ddc97",
            arrowprops=dict(arrowstyle="->", color="#3ddc97"), fontsize=10)
clock = ax.text(0.01, 0.95, "", transform=ax.transAxes, color="#e6edf7", fontsize=13, family="monospace")
# MN9 rate trace
ax2.set_xlim(0, T_MS); ax2.set_ylim(0, 1.05); ax2.set_yticks([])
ax2.set_xlabel("time (ms)", color="#8b98ad"); ax2.set_ylabel("MN9\nspikes", color="#3ddc97")
mn9_times = t_all[i_all == mn9]
trace, = ax2.plot([], [], color="#3ddc97", lw=1.5)
label = ax2.text(0.99, 0.8, "", transform=ax2.transAxes, color="#3ddc97", ha="right", fontsize=12)

def update(f):
    t = f * FRAME_MS
    sel = (t_all > t - GLOW_MS) & (t_all <= t)
    idx = np.array([pos[n] for n in i_all[sel]], dtype=int)
    age = (t - t_all[sel]) / GLOW_MS
    glow.set_offsets(np.c_[x[idx], y[idx]] if idx.size else np.empty((0, 2)))
    glow.set_sizes(60 * (1 - age) + 8); glow.set_array(1 - age)
    clock.set_text(f"t = {t:3d} ms   neurons firing now: {idx.size:3d}")
    m = mn9_times[mn9_times <= t]
    xs = np.repeat(m, 3); ys = np.tile([0, 1, 0], m.size)
    trace.set_data(xs, ys)
    label.set_text(f"MN9 fired {m.size} times" + ("  →  FLY IS FEEDING" if m.size > 3 else ""))
    return glow, clock, trace, label

anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
anim.save("brain_demo.gif", writer=PillowWriter(fps=25), dpi=80)
print("saved brain_demo.gif")
