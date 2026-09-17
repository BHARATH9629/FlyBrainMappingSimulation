"""Tune the reservoir hyper-parameters on TRAINING games, then re-run the fair 3-way comparison on unseen games."""
import os, json, time, itertools, numpy as np, scipy.sparse as sp
import flappy_fly as F

HERE = os.path.dirname(os.path.abspath(__file__))
Wfly_raw, fly_ids = F.load_subnetwork(3000)
Wrand_raw = F.shuffle_wiring(Wfly_raw)

def make(Wraw, rho, leak, in_scale, seed=0):
    return F.Reservoir(F.scale_spectral(Wraw, rho), F.Flappy.N_IN, seed=seed, leak=leak, in_scale=in_scale)

class WithInput:
    """Wrap a reservoir so the read-out also sees the 34 raw inputs (standard ESN trick)."""
    def __init__(self, res): self.res = res; self.x = None
    def reset(self): self.res.reset()
    def step(self, u): self.x = np.concatenate([self.res.step(u), u]); return self.x

def run(Wraw, cfg, frames=30000, val_seeds=(50, 51, 52, 53, 54)):
    res = make(Wraw, cfg["rho"], cfg["leak"], cfg["in_scale"])
    if cfg["input_to_readout"]: res = WithInput(res)
    X, Y = F.collect(res, frames, seed=1, noise=cfg["noise"])
    w = F.fit_readout(X, Y, cfg["lam"]); thr = F.calibrate(res, w, X)
    val = np.mean([F.play(res, w, s, thr=thr, max_frames=3000)[0] for s in val_seeds])
    return val, res, w, thr

grid = dict(rho=[0.8, 0.95, 1.1], leak=[0.3, 0.6, 0.9], in_scale=[1.0, 3.0], lam=[0.1, 3.0],
            input_to_readout=[False, True], noise=[0.15])
keys = list(grid)
rng = np.random.default_rng(0)
combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
rng.shuffle(combos); combos = combos[:24]                      # random subset of the grid
print(f"searching {len(combos)} configs on the FLY brain, validation = 5 training-side games\n")
best = (-1, None); t0 = time.time()
for i, cfg in enumerate(combos):
    val, *_ = run(Wfly_raw, cfg)
    print(f"[{i+1:2d}/{len(combos)}] {cfg}  ->  val {val:5.1f} pipes   ({time.time()-t0:.0f}s)")
    if val > best[0]: best = (val, cfg)
print("\nBEST config:", best[1], "val", best[0])
json.dump(best[1], open(os.path.join(HERE, "best_cfg.json"), "w"))

print("\nFair comparison on 10 UNSEEN games with the best config (same settings for all three):")
final = {}
for label, Wraw in (("fly brain", Wfly_raw), ("random brain", Wrand_raw)):
    _, res, w, thr = run(Wraw, best[1], frames=40000)
    final[label] = [F.play(res, w, 100 + i, thr=thr)[0] for i in range(10)]
    print(f"  {label:14s} {final[label]}  mean {np.mean(final[label]):.1f}")
    if label == "fly brain":
        np.save(os.path.join(HERE, "readout_fly.npy"), np.append(w, thr)); sp.save_npz(os.path.join(HERE, "W_fly.npz"), F.scale_spectral(Wfly_raw, best[1]["rho"]))
# no-brain baseline (linear on inputs) unchanged by these knobs except lam
X, Y = F.collect(None, 40000, seed=1); w = F.fit_readout(X, Y, best[1]["lam"]); thr = F.calibrate(None, w, X)
final["no brain"] = [F.play(None, w, 100 + i, thr=thr)[0] for i in range(10)]
print(f"  {'no brain':14s} {final['no brain']}  mean {np.mean(final['no brain']):.1f}")
json.dump(final, open(os.path.join(HERE, "final_scores.json"), "w"))
