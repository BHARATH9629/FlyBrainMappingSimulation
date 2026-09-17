"""Render the fly brain playing Flappy Bird: game on the left, the 3,000 real neurons on the right."""
import os, numpy as np, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import flappy_fly as F

HERE = os.path.dirname(os.path.abspath(__file__))
Wfly = sp.load_npz(os.path.join(HERE, "W_fly.npz"))
wt = np.load(os.path.join(HERE, "readout_fly.npy")); w, thr = wt[:-1], wt[-1]
res = F.Reservoir(Wfly, F.Flappy.N_IN)

# pick the best of a few games to show
best = max((F.play(res, w, seed=s, max_frames=900, record=True, thr=thr) for s in (101, 102, 104, 105, 107, 108, 109)), key=lambda r: r[0])
score, T, frames = best
print(f"showing a game with {score} pipes over {T} frames")
frames = frames[::2]                       # 2 game frames per video frame

# neuron layout: force-free random cloud; colour = activity
rng = np.random.default_rng(0); n = Wfly.shape[0]
ang = rng.uniform(0, 2 * np.pi, n); rad = np.sqrt(rng.uniform(0, 1, n))
nx, ny = rad * np.cos(ang), rad * np.sin(ang)

BG = "#0b0f17"
fig = plt.figure(figsize=(12, 6), facecolor=BG)
axg = fig.add_axes([0.04, 0.08, 0.42, 0.84]); axb = fig.add_axes([0.52, 0.08, 0.46, 0.84])
for a in (axg, axb): a.set_facecolor(BG); a.set_xticks([]); a.set_yticks([])
axg.set_xlim(0, 1); axg.set_ylim(1, 0)                     # y grows downward like the game
axg.set_title("Flappy Bird", color="#e6edf7", fontsize=13)
axb.set_xlim(-1.1, 1.1); axb.set_ylim(-1.1, 1.1)
axb.set_title("The player: 3,000 real fly neurons, 241,219 real synapses (untrained)", color="#e6edf7", fontsize=11)
pipe_patches = [Rectangle((0, 0), 0, 0, color="#3ddc97") for _ in range(8)]
for p in pipe_patches: axg.add_patch(p)
bird = Circle((F.Flappy.BIRD_X, 0.5), 0.022, color="#ffd966", zorder=5); axg.add_patch(bird)
score_txt = axg.text(0.02, 0.05, "", color="#e6edf7", fontsize=14, weight="bold")
flap_txt = axg.text(0.98, 0.05, "", color="#ff8c5c", fontsize=12, ha="right", weight="bold")
dots = axb.scatter(nx, ny, s=6, c=np.zeros(n), cmap="inferno", vmin=0, vmax=0.8)
cap = fig.text(0.5, 0.015, "Only a 3,000-number read-out was trained. The wiring is the fly's own. "
               "Same game with the wiring scrambled: 4 pipes on average; with real wiring: 11.",
               ha="center", color="#8b98ad", fontsize=10)

def update(i):
    y, pipes, sc, x, flap = frames[i]
    for p in pipe_patches: p.set_width(0)
    k = 0
    for px, g in pipes:
        if -0.05 < px < 1.05:
            pipe_patches[k].set_bounds(px - 0.03, 0, 0.06, g - F.Flappy.GAP / 2); k += 1
            pipe_patches[k].set_bounds(px - 0.03, g + F.Flappy.GAP / 2, 0.06, 1 - g - F.Flappy.GAP / 2); k += 1
    bird.center = (F.Flappy.BIRD_X, y)
    score_txt.set_text(f"pipes: {sc}"); flap_txt.set_text("FLAP" if flap else "")
    dots.set_array(np.abs(x))
    return pipe_patches + [bird, score_txt, flap_txt, dots]

anim = FuncAnimation(fig, update, frames=len(frames), interval=50)
anim.save(os.path.join(HERE, "flappy_fly_brain.gif"), writer=PillowWriter(fps=20), dpi=80)
print("saved flappy_fly_brain.gif")
