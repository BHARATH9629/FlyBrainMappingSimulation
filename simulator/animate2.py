"""Animation 2: a cartoon fly whose proboscis (mouth) is driven live by the simulated brain.
Timeline: 0 ms sugar arrives -> brain lights up -> MN9 fires -> proboscis extends.
At 350 ms we 'cut' the 3 command neurons -> MN9 goes quiet -> proboscis retracts."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse.csgraph import dijkstra
import sugar_test as st

ids, id2idx, W, N = st.ids, st.id2idx, st.W, st.N
mn9 = id2idx[st.MN9]
sugar = np.array([id2idx[i] for i in st.SUGAR if i in id2idx])
CUT = [id2idx[i] for i in (720575940623211725, 720575940627099338, 720575940632252743)]
T_MS, T_CUT = 700, 350

# ---- simulate with a lesion applied at T_CUT ----
Wl = W.tolil(copy=True); Wl[CUT, :] = 0; Wl = Wl.tocsr()
spikes_t, spikes_i = [], []
rng = np.random.default_rng(1)
v = np.full(N, st.V0, np.float32); g = np.zeros(N, np.float32); refr = np.zeros(N, np.float32)
rfc = np.full(N, st.T_RFC, np.float32); rfc[sugar] = 0
n_steps = int(T_MS / st.DT); n_delay = int(round(st.T_DLY / st.DT))
queue = [np.empty(0, np.int64) for _ in range(n_delay)]
p_poi = st.R_POI * st.DT / 1000; w_poi = st.W_SYN * st.F_POI; dv = st.DT / st.T_MBR; dg = np.float32(np.exp(-st.DT / st.TAU))
for step in range(n_steps):
    Wcur = W if step * st.DT < T_CUT else Wl
    arriving = queue.pop(0)
    if arriving.size: g += np.asarray(Wcur[arriving].sum(axis=0)).ravel()
    hit = sugar[rng.random(sugar.size) < p_poi]; v[hit] += w_poi
    act = refr <= 0
    v[act] += dv * (st.V0 - v[act] + g[act]); g[act] *= dg; refr -= st.DT
    spk = np.flatnonzero(v > st.V_TH)
    if spk.size:
        v[spk] = st.V_RST; g[spk] = 0; refr[spk] = rfc[spk]
        spikes_t.append(np.full(spk.size, step * st.DT)); spikes_i.append(spk)
    queue.append(spk)
t_all = np.concatenate(spikes_t); i_all = np.concatenate(spikes_i)
mn9_t = t_all[i_all == mn9]
print(f"MN9 spikes before cut: {(mn9_t < T_CUT).sum()}, after cut: {(mn9_t >= T_CUT).sum()}")

# ---- brain layout (hops from tongue) ----
active = np.unique(i_all)
adj = (W.multiply(W >= 3 * st.W_SYN) > 0).astype(np.int8)
hops = dijkstra(adj, directed=True, indices=sugar, unweighted=True, min_only=True, limit=6)
hops = np.where(np.isfinite(hops), hops, 6)
lr = np.random.default_rng(0)
bx = hops[active] + lr.uniform(-0.35, 0.35, active.size); by = lr.uniform(0, 1, active.size)
pos = {n: k for k, n in enumerate(active)}
bx[pos[mn9]], by[pos[mn9]] = hops[mn9], 0.5          # MN9 pinned exactly inside its circle
cut_pos = [pos[c] for c in CUT if c in pos]

# ---- figure ----
BG = "#0b0f17"
fig = plt.figure(figsize=(12, 6.2), facecolor=BG)
axb = fig.add_axes([0.05, 0.12, 0.55, 0.78]); axf = fig.add_axes([0.64, 0.12, 0.33, 0.78])
for a in (axb, axf): a.set_facecolor(BG)
axb.set_xlim(-0.6, 6.6); axb.set_ylim(-0.05, 1.05); axb.set_yticks([])
axb.set_xticks(range(7)); axb.set_xticklabels(["tongue", "1 hop", "2", "3", "4", "5", "6+"], color="#e6edf7")
axb.tick_params(colors="#8b98ad")
axb.set_title("Inside the brain (real wiring, 139,255 neurons)", color="#e6edf7", fontsize=12)
axb.scatter(bx, by, s=8, c="#233044")
glow = axb.scatter([], [], s=[], c=[], cmap="autumn", vmin=0, vmax=1)
axb.scatter([hops[mn9]], [0.5], s=420, facecolors="none", edgecolors="#3ddc97", lw=2.5, zorder=4)
mn9_flash = axb.scatter([hops[mn9]], [0.5], s=[0], c="#3ddc97", zorder=5)
axb.text(hops[mn9] + 0.2, 0.56, "MN9 = 'eat' command\n(flashes green when it fires)", color="#3ddc97", fontsize=9)
cut_marks = axb.scatter(bx[cut_pos], by[cut_pos], s=220, marker="x", c="#ff5c6c", lw=2.5, alpha=0)
clock = axb.text(0.01, 0.95, "", transform=axb.transAxes, color="#e6edf7", family="monospace", fontsize=12)
status = fig.text(0.5, 0.03, "", ha="center", color="#e6edf7", fontsize=13, weight="bold")

# cartoon fly (side view, head on the right)
axf.set_xlim(0, 10); axf.set_ylim(0, 10); axf.axis("off")
axf.set_title("The fly", color="#e6edf7", fontsize=12)
axf.add_patch(Ellipse((4.2, 5.2), 4.6, 2.4, color="#3b3b4a"))                      # abdomen+thorax
axf.add_patch(Ellipse((4.0, 5.2), 4.4, 2.2, color="#4a4a5e"))
for i, xx in enumerate((2.6, 3.4, 4.2)):                                            # stripes
    axf.add_patch(Ellipse((xx, 5.2), 0.5, 2.0, color="#2c2c38"))
axf.add_patch(Ellipse((4.6, 6.7), 5.0, 1.3, angle=-12, color="#9fb3c8", alpha=0.35)) # wing
head = Circle((7.0, 5.4), 1.05, color="#4a4a5e"); axf.add_patch(head)
axf.add_patch(Ellipse((7.45, 5.7), 0.9, 1.0, color="#b3202a"))                       # eye
axf.add_patch(Circle((7.6, 5.85), 0.18, color="#ff8080"))
for x0, x1 in ((3.0, 2.2), (4.2, 4.0), (5.6, 6.4)):                                  # legs
    axf.plot([x0, x1], [4.2, 2.6], color="#3b3b4a", lw=3)
axf.plot([7.4, 8.6], [6.2, 7.6], color="#3b3b4a", lw=2)                             # antenna
prob, = axf.plot([7.3, 7.3], [4.6, 4.6], color="#c9a86a", lw=6, solid_capstyle="round")  # proboscis
sugar_drop = Circle((7.6, 1.6), 0.55, color="#ffd966"); axf.add_patch(sugar_drop)
axf.text(7.6, 0.7, "sugar", ha="center", color="#ffd966", fontsize=10)
flabel = axf.text(5, 9.2, "", ha="center", color="#3ddc97", fontsize=13, weight="bold")

FRAME_MS, GLOW_MS = 5, 12
frames = int(T_MS / FRAME_MS)
def update(f):
    t = f * FRAME_MS
    sel = (t_all > t - GLOW_MS) & (t_all <= t)
    idx = np.array([pos[n] for n in i_all[sel]], dtype=int)
    age = (t - t_all[sel]) / GLOW_MS
    glow.set_offsets(np.c_[bx[idx], by[idx]] if idx.size else np.empty((0, 2)))
    glow.set_sizes(60 * (1 - age) + 8); glow.set_array(1 - age)
    # proboscis length follows MN9 firing in the last 60 ms
    rate = ((mn9_t > t - 60) & (mn9_t <= t)).sum()
    recent = mn9_t[(mn9_t > t - 40) & (mn9_t <= t)]
    mn9_flash.set_sizes([0] if recent.size == 0 else [380 * (1 - (t - recent.max()) / 40) + 60])
    L = min(2.6, 0.45 * rate)
    prob.set_data([7.3, 7.35], [4.6, 4.6 - L])
    clock.set_text(f"t = {t:3d} ms   firing now: {idx.size:3d}")
    if t < T_CUT:
        status.set_text("Sugar on the tongue  ->  brain decides  ->  fly extends its mouth to eat")
        status.set_color("#3ddc97"); cut_marks.set_alpha(0)
        flabel.set_text("FEEDING" if rate >= 3 else "")
    else:
        status.set_text(f"BRAIN SURGERY at {T_CUT} ms: 3 command neurons cut (red X)  ->  feeding stops")
        status.set_color("#ff5c6c"); cut_marks.set_alpha(1)
        flabel.set_text("FEEDING" if rate >= 3 else "stopped"); flabel.set_color("#3ddc97" if rate >= 3 else "#ff5c6c")
    return glow, prob, clock, status, flabel, cut_marks, mn9_flash

anim = FuncAnimation(fig, update, frames=frames, interval=40)
anim.save("fly_feeding_demo.gif", writer=PillowWriter(fps=25), dpi=80)
print("saved fly_feeding_demo.gif")
