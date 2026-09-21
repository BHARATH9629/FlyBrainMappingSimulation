"""Choke-point map: which neurons appear in (almost) every minimal attack on a behaviour, and does wiring topology
predict them?

1. Enumerate minimal BLOCK (or corrupt) attacks: disjoint sets from redundancy(), then greedy again with each neuron of
   every found set excluded in turn (and, with --search, the beam search too). Every set is minimised by the agents.
2. Per neuron: frequency = fraction of minimal attacks that use it.
3. Topology scores on the stimulus-reachable sub-network: direct synapses onto the readout, stimulus->neuron drive,
   and shortest-path betweenness restricted to stimulus->readout paths. Spearman rank correlation with frequency.
4. Writes atlas/<sex>/chokepoints_<behaviour>.csv and .png (dots coloured by frequency on the brain).
Usage: python chokepoints.py --sex female --behaviour eat [--search] [--max_sets 10]
"""
import argparse, csv, json, os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from brain import Brain
from behaviours import Behaviour, RULES
from agents import Attacker, Log, redundancy, CHECK_SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--max_sets", type=int, default=10); ap.add_argument("--search", action="store_true")
ap.add_argument("--no_png", action="store_true")
args = ap.parse_args()

brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); log = Log(quiet=True)
goal = "corrupt" if b.expected == "off" else "block"
t0 = time.time()


def neurons_of(atk):
    iv = atk["intervention"]
    return frozenset(set(iv.get("lesion", [])) | set(iv.get("inject", {})) | set(iv.get("weaken", {})))


# ---------------------------------------------------------------- 1. enumerate minimal attacks
STRATS = ("greedy",) if goal == "corrupt" else ("greedy", "inject_inhibitors", "weaken_only")
def attacks_excluding(excl, tag):
    out = []
    for strat in STRATS:
        atk = getattr(Attacker(b, log, goal=goal, excluded=excl), strat)(name=f"{tag}-{strat}")
        if atk: out.append(atk)
    if args.search:
        atk = Attacker(b, log, goal=goal, excluded=excl).search(name=f"{tag}-search", max_size=RULES["moves"]["lesion"]["max"])
        if atk: out.append(atk)
    return out

sets = {}
for atk in redundancy(b, log, goal=goal, max_sets=args.max_sets):
    sets[neurons_of(atk)] = atk
for atk in attacks_excluding(set(), "all"):
    sets.setdefault(neurons_of(atk), atk)
print(f"{args.sex} {b.name}: {len(sets)} attacks from redundancy + strategies (sizes {[len(s) for s in sets]})", flush=True)
frontier = list(sets)
seen_excl = set()
for _ in range(3):                                            # a few rounds of "exclude one neuron, search again"
    new = []
    for S in frontier:
        for n in S:
            if n in seen_excl: continue
            seen_excl.add(n)
            for atk in attacks_excluding({n}, f"excl-{n}"):
                key = neurons_of(atk)
                if key not in sets: sets[key] = atk; new.append(key)
    frontier = new
    if not new: break
    print(f"  round: {len(new)} new minimal attacks, {len(sets)} total  ({time.time()-t0:.0f}s)", flush=True)

# keep only sets that are minimal w.r.t. the collection (no found set is a strict subset)
keys = [k for k in sets if not any(o < k for o in sets)]
print(f"  {len(keys)} minimal attacks kept, sizes {sorted(len(k) for k in keys)}")
freq = {}
for k in keys:
    for n in k: freq[n] = freq.get(n, 0) + 1
freq = {n: c / len(keys) for n, c in freq.items()}

# ---------------------------------------------------------------- 3. topology on the reachable sub-network
sub = brain.subnetwork(b.stim, depth=b.sim.get("depth", 3), min_syn=b.sim.get("min_syn", 5), must_include=b.readouts)
loc = {int(g): i for i, g in enumerate(sub)}; n = sub.size
Wl = brain.W[sub][:, sub].tocsr(); A = (abs(Wl) > 0).astype(np.int8)
ro = [loc[r] for r in b.readouts if r in loc]; st = [loc[s] for s in b.stim if s in loc]
_, r_on = b.measure(True); active = [i for i in r_on["active"] if i in loc]
# direct synapses onto the readout (signed)
col = np.asarray(brain.W[:, b.readouts].sum(axis=1)).ravel()
# stimulus -> neuron direct drive
drive = np.asarray(abs(brain.W[b.stim]).sum(axis=0)).ravel()
# shortest-path betweenness restricted to stimulus -> readout: neuron i is on a shortest path iff d_s[i] + d_r[i] == D
from scipy.sparse.csgraph import shortest_path
d_s = shortest_path(A, directed=True, unweighted=True, indices=st).min(axis=0)
d_r = shortest_path(A.T, directed=True, unweighted=True, indices=ro).min(axis=0)
D = d_r[st].min() if len(st) else np.inf
print(f"  shortest stimulus->readout path: {D:.0f} hops; sub-network {n:,} neurons")
# path counts: forward counts from stimulus, backward counts to readout (layered DAG of shortest paths)
def path_counts(dist, sources, Adj):
    cnt = np.zeros(n); cnt[sources] = 1.0
    order = np.argsort(dist); Adj = Adj.tocsr()
    for lvl in range(1, int(np.nanmax(dist[np.isfinite(dist)])) + 1):
        prev = np.flatnonzero(dist == lvl - 1); cur_mask = dist == lvl
        if prev.size == 0: break
        contrib = np.asarray(Adj[prev].T @ cnt[prev]).ravel()
        cnt[cur_mask] = contrib[cur_mask]
    return cnt
fwd = path_counts(d_s, st, A); bwd = path_counts(d_r, ro, A.T)
on_path = np.isfinite(d_s) & np.isfinite(d_r) & (d_s + d_r == D)
betw = np.where(on_path, fwd * bwd, 0.0); total = betw[ro].sum() if len(ro) else 1.0
betw = betw / max(total, 1.0)

rows = []
for i in sorted(set(active) | set(freq)):
    j = loc.get(i)
    rows.append(dict(index=i, id=int(brain.ids[i]), type=brain.type_of(i), frequency=round(freq.get(i, 0.0), 3),
                     in_attacks=int(round(freq.get(i, 0.0) * len(keys))), rate_hz=round(r_on["active"].get(i, 0.0), 1),
                     syn_to_readout=float(col[i]), stim_drive=float(drive[i]),
                     sp_betweenness=float(betw[j]) if j is not None else 0.0))
os.makedirs(f"atlas/{args.sex}", exist_ok=True)
out = f"atlas/{args.sex}/chokepoints_{b.name}"
with open(out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: -r["frequency"]))

# correlations over the active neurons (the attack candidates)
cand = [r for r in rows if r["rate_hz"] > 0]
f_ = np.array([r["frequency"] for r in cand])
corr = {}
for name in ("syn_to_readout", "stim_drive", "sp_betweenness", "rate_hz"):
    x = np.array([abs(r[name]) for r in cand])
    rho, p = spearmanr(x, f_) if len(cand) > 3 else (np.nan, np.nan)
    corr[name] = (float(rho), float(p))
top = sorted(rows, key=lambda r: -r["frequency"])[:12]
print(f"\nCHOKE POINTS ({goal}, {len(keys)} minimal attacks):")
for r in top:
    if r["frequency"] == 0: break
    print(f"  {r['index']:>7d} {r['type'] or '?':14s} in {r['in_attacks']:2d}/{len(keys)} attacks  rate {r['rate_hz']:6.1f} Hz  "
          f"syn->readout {r['syn_to_readout']:6.0f}  betweenness {r['sp_betweenness']:.3f}")
print("Spearman rho (attack frequency vs topology, over active neurons): " +
      ", ".join(f"{k} {v[0]:+.2f} (p={v[1]:.2g})" for k, v in corr.items()))
json.dump({"sex": args.sex, "behaviour": b.name, "goal": goal, "n_minimal_attacks": len(keys),
           "sizes": sorted(len(k) for k in keys), "attacks": [sorted(k) for k in keys], "spearman": corr,
           "seconds": round(time.time() - t0)}, open(out + ".json", "w"), indent=1)
print(f"saved {out}.csv / .json  ({time.time()-t0:.0f}s)")

if not args.no_png:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "virtual_fly"))
    from brainview import BrainView
    import imageio.v2 as iio
    v = BrainView(brain, 960, 720)
    img = v.base.copy()
    for r in rows:
        if r["frequency"] > 0:
            c = r["frequency"]
            v._splat(img, np.array([r["index"]]), (1.0, 0.2 + 0.6 * (1 - c), 0.1), size=2.0 + 4.0 * c)
    v._splat(img, np.array([i for i in b.stim if v.ok[i]]), (0.3, 0.9, 1.0), size=2.0)
    for ro_i in b.readouts: v._splat(img, np.array([ro_i]), (0.2, 1.0, 0.3), size=4.0)
    from PIL import Image, ImageDraw
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.text((10, 8), f"{args.sex} fly, {b.name}: choke points of {len(keys)} minimal {goal} attacks (red = in every attack, orange = in some)", fill=(235, 235, 235), font=v.font)
    d.text((10, 28), "cyan = stimulus, green = readout", fill=(180, 180, 190), font=v.font)
    im.save(out + ".png"); print(f"saved {out}.png")
