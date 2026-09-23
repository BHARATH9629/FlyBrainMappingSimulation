"""Do the conclusions survive the model's arbitrary choices?

Every result so far uses one set of LIF parameters (Shiu et al. 2024) and one neurotransmitter sign per neuron. Both
are choices, and the sign in particular is a machine prediction that is wrong for some cells. This script re-measures
the two headline claims under each perturbed model:

  claim A (blocking is hard)  : a random set of k neurons almost never blocks the behaviour, while the red team's set does
  claim B (forcing is easy)   : a single randomly driven neuron forces it a noticeable fraction of the time

If the asymmetry between A and B survives every variant, it is a property of the wiring. If it flips, it is an artefact
of our settings -- and we would rather find that here than in review.

Usage: python robustness.py --sex female --behaviour eat --draws 25
"""
import argparse, json, os, time
import numpy as np
import brain as brain_mod
from brain import Brain
from behaviours import Behaviour
from agents import Attacker, Log

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--draws", type=int, default=25, help="random sets drawn per condition, per claim")
ap.add_argument("--seeds", type=int, default=2, help="seeds each candidate must satisfy")
ap.add_argument("--sign_flip", type=float, nargs="*", default=[0.05, 0.10], help="fractions of neurons whose transmitter sign is flipped")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); out_dir = os.path.join(HERE, "atlas", args.sex)

DEFAULTS = dict(W_SYN=brain_mod.W_SYN, V_TH=brain_mod.V_TH, T_MBR=brain_mod.T_MBR, TAU=brain_mod.TAU, T_RFC=brain_mod.T_RFC)
brain = Brain(args.sex)
W0 = brain.W.copy()                                   # the untouched signed synapse counts


def set_model(**kw):
    """apply LIF overrides (or restore defaults) and clear every cache that depends on them"""
    for k, v in DEFAULTS.items(): setattr(brain_mod, k, kw.get(k, v))
    brain._sub_cache.clear(); brain._run_cache.clear()


def flip_signs(fraction, rng):
    """flip the transmitter sign of a random `fraction` of neurons (all of their outgoing synapses)"""
    brain.W = W0.copy()
    if fraction:
        flip = rng.random(brain.N) < fraction
        d = np.ones(brain.N, np.float32); d[flip] = -1.0
        import scipy.sparse as sp
        brain.W = (sp.diags(d).astype(np.float32) @ brain.W).tocsr()
    brain._sub_cache.clear(); brain._run_cache.clear()
    return int(np.sum(brain.W.data * W0.data < 0)) if fraction else 0


def evaluate(label, note=""):
    """measure both claims on the brain as it currently stands"""
    b = Behaviour(brain, args.behaviour)
    seeds = tuple(range(1, args.seeds + 1))
    on = float(np.mean([b.measure(True, seed=s)[0] for s in seeds])); off = float(np.mean([b.measure(False, seed=s)[0] for s in seeds]))
    healthy_ok = on >= b.on_thr and off <= b.off_thr
    row = dict(condition=label, note=note, healthy_on_hz=round(on, 1), healthy_off_hz=round(off, 1), healthy_passes=healthy_ok)
    if not healthy_ok:                                 # the behaviour itself is gone: nothing to attack
        row.update(red_team_blocks=None, random_block_hits=None, random_force_hits=None)
        print(f"  {label:26s} on {on:6.1f} off {off:5.1f}  -- behaviour does not pass, skipped", flush=True); return row

    _, r0 = b.measure(True)
    pool = sorted(int(i) for i in r0["active"] if i not in b.protected)
    rng = np.random.default_rng(0)
    atk_block = Attacker(b, Log(quiet=True), goal="block", seeds=seeds)
    atk_force = Attacker(b, Log(quiet=True), goal="force", seeds=seeds)

    ok_red, hz_red = atk_block._check({"lesion": RED_TEAM_CUT})          # the attack found on the default model
    hits_b = sum(atk_block._check({"lesion": [int(x) for x in rng.choice(pool, len(RED_TEAM_CUT), replace=False)]})[0] for _ in range(args.draws))
    hits_f = sum(atk_force._check({"inject": {int(rng.choice(pool)): 150.0}})[0] for _ in range(args.draws))
    row.update(red_team_blocks=bool(ok_red), red_team_hz=round(hz_red, 1), pool=len(pool),
               random_block_hits=int(hits_b), random_force_hits=int(hits_f), draws=args.draws)
    print(f"  {label:26s} on {on:6.1f} off {off:5.1f} | red team {'blocks' if ok_red else 'FAILS'} ({hz_red:.0f} Hz) | "
          f"random blocks {hits_b}/{args.draws}, random forces {hits_f}/{args.draws}", flush=True)
    return row


# the red team's own attack on the default model, from the atlas
e = json.load(open(os.path.join(out_dir, f"{args.behaviour}.json")))["entry"]["goals"]["block"]["attacks"]
RED_TEAM_CUT = min((a["intervention"]["lesion"] for a in e.values() if a and a.get("intervention", {}).get("lesion")), key=len)
print(f"{args.sex} {args.behaviour}: red team's cut = {RED_TEAM_CUT}; {args.draws} random draws per claim, "
      f"goals checked on seeds 1-{args.seeds}\n", flush=True)

rows = []; t0 = time.time()
set_model(); flip_signs(0.0, np.random.default_rng(0))
rows.append(evaluate("baseline (published)", "Shiu et al. 2024 parameters, connectome signs as given"))

for name, kw, note in (
    ("synapse weight -30 %", dict(W_SYN=DEFAULTS["W_SYN"] * 0.7), "weaker synapses"),
    ("synapse weight +30 %", dict(W_SYN=DEFAULTS["W_SYN"] * 1.3), "stronger synapses"),
    ("threshold -1 mV", dict(V_TH=DEFAULTS["V_TH"] - 1.0), "neurons more excitable"),
    ("threshold +1 mV", dict(V_TH=DEFAULTS["V_TH"] + 1.0), "neurons less excitable"),
    ("membrane tau 15 ms", dict(T_MBR=15.0), "faster membrane"),
    ("membrane tau 25 ms", dict(T_MBR=25.0), "slower membrane"),
    ("refractory 1.5 ms", dict(T_RFC=1.5), "higher maximum firing rate"),
):
    set_model(**kw); rows.append(evaluate(name, note))

set_model()
for frac in args.sign_flip:
    for rep in range(2):
        flip_signs(frac, np.random.default_rng(100 + rep))
        rows.append(evaluate(f"{int(frac*100)} % signs flipped #{rep+1}",
                             "excitatory/inhibitory prediction wrong for this fraction of neurons"))
flip_signs(0.0, np.random.default_rng(0)); set_model()

# ---- verdict ----------------------------------------------------------------
usable = [r for r in rows if r["healthy_passes"]]
kept = [r for r in usable if r["red_team_blocks"]]
asym = [r for r in usable if r["random_force_hits"] > r["random_block_hits"]]
print(f"\n{len(usable)}/{len(rows)} conditions keep the behaviour itself intact")
print(f"  red team's original cut still blocks in {len(kept)}/{len(usable)}")
print(f"  forcing easier than blocking (random hits) in {len(asym)}/{len(usable)}")
print(f"  ({time.time()-t0:.0f}s)")
json.dump(dict(sex=args.sex, behaviour=args.behaviour, draws=args.draws, seeds=args.seeds,
               red_team_cut=RED_TEAM_CUT, conditions=rows), open(os.path.join(out_dir, f"robustness_{args.behaviour}.json"), "w"), indent=1)
print(f"saved atlas/{args.sex}/robustness_{args.behaviour}.json")
