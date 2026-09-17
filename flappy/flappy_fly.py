"""
Flappy Bird played by the fruit-fly connectome.

How it works
------------
1. Take a 3,000-neuron sub-network of the real FlyWire brain (signed synapse counts kept:
   excitatory > 0, inhibitory < 0).  This is the AI's "hidden layer" and is NEVER trained.
2. The game is shown to the brain through a crude 20x12 pixel retina (240 binary pixels),
   each pixel wired to a few random input neurons.
3. Only a linear read-out (3,000 -> 1 number: flap / don't flap) is trained, by imitation:
   we watch a simple rule-based "coach" play for a while, record what the brain looked like
   at each frame, and fit the read-out with ridge regression.
4. Then the brain plays on its own.  The same pipeline is run with the fly's wiring randomly
   shuffled (same size, same weights, wrong places) and with no brain at all (linear on pixels),
   so we can compare: fly brain vs random brain vs no brain.
"""
import os, time, argparse, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ============================================================ game
class Flappy:
    W, H = 20, 12                 # retina resolution (columns x rows)
    BLIND = 0.0      # fraction of the approach during which the pipe is INVISIBLE (needs memory)
    NOISE = 0.0      # probability each input bit is flipped (needs temporal averaging)
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed); self.reset()
    def reset(self):
        self.y, self.vy, self.t, self.score, self.alive = 0.5, 0.0, 0, 0, True
        self.pipes = [(1.2 + i * 0.7, self.rng.uniform(0.3, 0.7)) for i in range(4)]   # (x, gap centre)
        return self.retina()
    GAP, GRAV, FLAP, SPEED, BIRD_X = 0.30, 0.0022, -0.028, 0.012, 0.25
    def step(self, flap):
        if flap: self.vy = self.FLAP
        self.vy += self.GRAV; self.y += self.vy; self.t += 1
        self.pipes = [(x - self.SPEED, g) for x, g in self.pipes]
        if self.pipes[0][0] < -0.1:
            self.pipes.pop(0); self.pipes.append((self.pipes[-1][0] + 0.7, self.rng.uniform(0.3, 0.7))); self.score += 1
        x, g = self.next_pipe()
        hit = abs(x - self.BIRD_X) < 0.06 and abs(self.y - g) > self.GAP / 2
        if hit or self.y < 0 or self.y > 1: self.alive = False
        return self.retina(), self.alive
    def next_pipe(self):
        return next(p for p in self.pipes if p[0] > self.BIRD_X - 0.06)
    N_IN = 12 + 12 + 10
    def retina(self):
        """What the brain sees each frame (34 on/off inputs): my height (12 rows), the next gap's height
        (12 rows), distance to the next pipe (10 bins). NO velocity - the brain must infer motion from memory."""
        x, g = self.next_pipe()
        v = np.zeros(self.N_IN, np.float32)
        v[min(11, max(0, int(self.y * 12)))] = 1
        d = (x - self.BIRD_X) / 0.7                      # 1 = far, 0 = at the pipe
        if d > self.BLIND:                               # gap visible only while far enough away
            v[12 + min(11, max(0, int(g * 12)))] = 1
        v[24 + min(9, max(0, int(d * 10)))] = 1
        if self.NOISE > 0:
            flip = self.rng.random(self.N_IN) < self.NOISE; v[flip] = 1 - v[flip]
        return v
    def coach(self):
        """Rule-based teacher: flap when below the gap centre and not already rising fast."""
        x, g = self.next_pipe()
        return float(self.y > g + 0.02 and self.vy > -0.012)

# ============================================================ brain
def load_subnetwork(n_neurons=3000, seed=0):
    comp = pd.read_csv(os.path.join(DATA, "Completeness_783.csv"), index_col=0)
    con = pd.read_parquet(os.path.join(DATA, "Connectivity_783.parquet"),
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    N = len(comp)
    W = sp.csr_matrix((con["Excitatory x Connectivity"].values.astype(np.float32),
                       (con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values)), shape=(N, N))
    # grow a connected ball of neurons around a random seed, following strong connections (>=5 synapses)
    strong = (abs(W) >= 5).astype(np.int8)
    und = ((strong + strong.T) > 0).astype(np.int8).tocsr()          # grow along either direction
    deg = np.asarray(und.sum(axis=1)).ravel()
    rng = np.random.default_rng(seed)
    frontier = [int(np.argsort(deg)[-50:][rng.integers(50)])]        # start from one of the 50 biggest hubs
    chosen = set(frontier)
    while len(chosen) < n_neurons and frontier:
        nxt = []
        for i in frontier:
            for j in und[i].indices:
                if j not in chosen:
                    chosen.add(j); nxt.append(j)
                    if len(chosen) >= n_neurons: break
            if len(chosen) >= n_neurons: break
        frontier = nxt
    idx = np.array(sorted(chosen))
    Wsub = W[idx][:, idx].astype(np.float64)
    return Wsub, comp.index.values[idx]

def scale_spectral(Wsub, rho=0.95):
    lam = abs(eigs(Wsub.T.tocsc(), k=1, return_eigenvectors=False, which="LM", tol=1e-3)[0])
    return (Wsub * (rho / lam)).tocsr()

def shuffle_wiring(Wsub, seed=0):
    """Same neurons, same weight values, same number of connections - but random targets."""
    rng = np.random.default_rng(seed)
    coo = Wsub.tocoo(); n = Wsub.shape[0]
    return sp.csr_matrix((coo.data, (rng.integers(0, n, coo.nnz), rng.integers(0, n, coo.nnz))), shape=(n, n))

class Reservoir:
    """Rate-based echo-state network using the (signed) fly wiring as the recurrent matrix."""
    def __init__(self, Wsub, n_in, seed=0, leak=0.6, in_scale=2.0):
        rng = np.random.default_rng(seed)
        self.W = Wsub.T.tocsr(); n = Wsub.shape[0]          # W[post, pre]
        self.Win = sp.random(n, n_in, density=12 / n, random_state=seed, data_rvs=lambda k: rng.choice([-1, 1], k)).tocsr() * in_scale
        self.b = rng.normal(0, 0.1, n); self.leak = leak; self.x = np.zeros(n)
    def reset(self): self.x[:] = 0
    def step(self, u):
        pre = self.W @ self.x + self.Win @ u + self.b
        self.x = (1 - self.leak) * self.x + self.leak * np.tanh(pre)
        return self.x

class WithInput:
    """Read-out also sees the 34 raw inputs (standard echo-state-network trick)."""
    def __init__(self, res): self.res = res; self.x = None; self.W = res.W
    def reset(self): self.res.reset(); self.x = None
    def step(self, u): self.x = np.concatenate([self.res.step(u), u]); return self.x

DEFAULT_CFG = dict(rho=1.1, leak=0.6, in_scale=1.0, lam=0.1, input_to_readout=True, noise=0.15)
def load_cfg():
    p = os.path.join(HERE, "best_cfg.json")
    return {**DEFAULT_CFG, **json.load(open(p))} if os.path.exists(p) else DEFAULT_CFG

def build(kind, cfg, Wfly_raw):
    """kind: 'fly' | 'random' | 'none' -> a player object (or None for the linear baseline)."""
    if kind == "none": return None
    Wraw = Wfly_raw if kind == "fly" else shuffle_wiring(Wfly_raw)
    res = Reservoir(scale_spectral(Wraw, cfg["rho"]), Flappy.N_IN, leak=cfg["leak"], in_scale=cfg["in_scale"])
    return WithInput(res) if cfg["input_to_readout"] else res

# ============================================================ train + play
def collect(res, n_frames, seed, noise=0.15):
    """Watch the coach play (with a little randomness so we see recoveries). Record brain state + coach action."""
    game = Flappy(seed); rng = np.random.default_rng(seed)
    X, Y = [], []; obs = game.reset(); res.reset() if res else None
    while len(X) < n_frames:
        state = res.step(obs) if res is not None else obs
        a = game.coach()
        X.append(np.append(state, 1.0)); Y.append(a)
        act = a if rng.random() > noise else float(rng.random() < 0.15)
        obs, alive = game.step(act)
        if not alive: obs = game.reset(); (res.reset() if res else None)
    return np.array(X), np.array(Y)

def fit_readout(X, Y, lam=1.0):
    """Ridge regression to +1 (flap) / -1 (no flap), with classes re-weighted so the rare 'flap' counts equally."""
    sw = np.where(Y > 0.5, 0.5 / max(Y.mean(), 1e-6), 0.5 / max(1 - Y.mean(), 1e-6))
    Xw = X * sw[:, None]
    return np.linalg.solve(X.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ (2 * Y - 1))

def dagger(res, cfg, frames=40000, rounds=3, seed=1):
    """Imitation learning that also learns from the student's own mistakes (DAgger)."""
    X, Y = collect(res, frames, seed=seed, noise=cfg["noise"])
    w = fit_readout(X, Y, cfg["lam"]); thr = calibrate(res, w, X)
    for r in range(rounds):
        Xs, Ys = [], []
        for g in range(8):                                   # student plays, coach labels every frame it visits
            game = Flappy(1000 * (r + 1) + g); obs = game.reset(); res.reset() if res else None
            while game.alive and game.t < 1500:
                state = res.step(obs) if res is not None else obs
                Xs.append(np.append(state, 1.0)); Ys.append(game.coach())
                obs, _ = game.step((Xs[-1] @ w) > thr)
        X = np.vstack([X, np.array(Xs)]); Y = np.concatenate([Y, np.array(Ys)])
        w = fit_readout(X, Y, cfg["lam"]); thr = calibrate(res, w, X)
    return w, thr

def calibrate(res, w, X, seeds=(1, 2, 3)):
    """Pick the decision threshold that plays best on TRAINING games (one scalar; test games stay unseen)."""
    scores = X @ w; best = (-1, 0.0)
    for q in (0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99):
        thr = np.quantile(scores, q)
        sc = np.mean([play(res, w, s, thr=thr, max_frames=1500)[0] for s in seeds])
        if sc > best[0]: best = (sc, thr)
    return best[1]

def play(res, w, seed, max_frames=4000, record=False, thr=0.0):
    game = Flappy(seed); obs = game.reset(); res.reset() if res else None
    frames = []
    while game.alive and game.t < max_frames:
        state = res.step(obs) if res is not None else obs
        flap = (np.append(state, 1.0) @ w) > thr
        if record: frames.append((game.y, [p for p in game.pipes], game.score, res.x.copy() if res else None, bool(flap)))
        obs, _ = game.step(flap)
    return game.score, game.t, frames

def evaluate(res, w, n=10, label="", thr=0.0):
    scores = [play(res, w, seed=100 + i, thr=thr)[0] for i in range(n)]
    print(f"  {label:14s} pipes passed per game: {scores}  ->  mean {np.mean(scores):.1f}")
    return scores

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--neurons", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=40000); ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--blind", type=float, default=0.0, help="fraction of approach with pipe invisible, e.g. 0.5")
    ap.add_argument("--noise", type=float, default=0.0, help="probability of flipping each input bit")
    args = ap.parse_args(); cfg = load_cfg(); t0 = time.time()
    Flappy.BLIND, Flappy.NOISE = args.blind, args.noise
    print(f"task: blind={args.blind}  noise={args.noise}")
    Wfly_raw, fly_ids = load_subnetwork(args.neurons)
    print(f"fly sub-network: {Wfly_raw.shape[0]} neurons, {Wfly_raw.nnz:,} synapses "
          f"({(Wfly_raw.data > 0).mean() * 100:.0f}% excitatory)   config: {cfg}")
    np.save(os.path.join(HERE, "fly_ids.npy"), fly_ids); sp.save_npz(os.path.join(HERE, "W_fly_raw.npz"), Wfly_raw.tocsr())
    results = {}
    for kind in ("fly", "random", "none"):
        res = build(kind, cfg, Wfly_raw)
        w, thr = dagger(res, cfg, frames=args.frames)
        np.save(os.path.join(HERE, f"readout_{kind}.npy"), np.append(w, thr))
        scores = [play(res, w, seed=100 + i, thr=thr)[0] for i in range(args.games)]
        results[kind] = scores
        print(f"  {kind:7s} brain: mean {np.mean(scores):5.1f}  median {np.median(scores):4.0f}  "
              f"max {max(scores):3d}  zero-games {sum(s == 0 for s in scores)}/{args.games}   [{time.time()-t0:.0f}s]")
    json.dump(results, open(os.path.join(HERE, f"final_scores_blind{args.blind}_noise{args.noise}.json"), "w"))
    print("Summary (mean pipes over %d unseen games): " % args.games +
          "  ".join(f"{k}={np.mean(v):.1f}" for k, v in results.items()))
