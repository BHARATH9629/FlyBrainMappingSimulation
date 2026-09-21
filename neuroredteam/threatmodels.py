"""Does the attacker win because of the circuit, or because of the rules?  The iterated game under several threat
models, same attacker strategies (greedy / inject inhibitors / weaken) and the same robust defender.

  T0  current rules (control)
  T1  inject capped at --cap_hz (default 50 Hz): no super-physiological drive of inhibitory interneurons
  T2  the attacker may not touch the readout's DIRECT inputs (extends the 'never touch the readout' rule one hop)
  T3  current rules, but the defender may also bias the READOUT itself by +--readout_mv (default 2 mV)
Prints one row per threat model: single-shot minimal attack, game winner, costs, exchange ratio.
Usage: python threatmodels.py --sex female --behaviour eat [--models T0 T1 T2 T3] [--budget 12] [--rounds 8]
"""
import argparse, json, os, time
import numpy as np
from brain import Brain
from behaviours import Behaviour, RULES, DEFENCE_RULES
from agents import Attacker, Defender, Log, merge_iv, size_of

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--models", nargs="+", default=["T0", "T1", "T2", "T3"]); ap.add_argument("--budget", type=int, default=12)
ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--cap_hz", type=float, default=50.0)
ap.add_argument("--readout_mv", type=float, default=2.0)
args = ap.parse_args()

brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); log = Log(quiet=True); t0 = time.time()
ro = b.readout
direct_inputs = set(int(i) for i in np.flatnonzero(brain.W[:, ro].toarray().ravel() != 0))
DEFAULT_HZ = RULES["moves"]["inject"]["max_hz"]


def attack(fixed, excluded, tag):
    found = [a for a in (Attacker(b, log, "block", fixed=fixed, excluded=excluded).greedy(name=f"{tag}-greedy"),
                         Attacker(b, log, "block", fixed=fixed, excluded=excluded).inject_inhibitors(name=f"{tag}-inject"),
                         Attacker(b, log, "block", fixed=fixed, excluded=excluded).weaken_only(name=f"{tag}-weaken")) if a]
    return min(found, key=lambda a: (a["size"], a["simulations"])) if found else None


def play(model):
    """Set the rules for `model`, play the game, restore the rules."""
    RULES["moves"]["inject"]["max_hz"] = args.cap_hz if model == "T1" else DEFAULT_HZ
    DEFENCE_RULES["readout_bias_mv"] = args.readout_mv if model == "T3" else 0.0
    excluded = direct_inputs if model == "T2" else set()
    base = attack({}, excluded, f"{model}-base")
    fixed = {}; atk_cost = def_cost = 0; winner = None; rounds = []
    for r in range(1, args.rounds + 1):
        atk = attack(fixed, excluded, f"{model}-r{r}")
        if not atk: winner = "defender (attacker finds nothing)"; break
        if atk_cost + atk["size"] > args.budget: winner = "defender (attacker out of budget)"; break
        atk_cost += atk["size"]; fixed = merge_iv(fixed, atk["intervention"])
        rep = Defender(b, {"intervention": fixed}, log).robust(name=f"{model}-r{r}-def")
        if not rep: winner = "attacker (no repair)"; rounds.append(dict(round=r, attack=atk["size"], repair=None)); break
        if def_cost + rep["size"] > args.budget: winner = "attacker (defender out of budget)"; rounds.append(dict(round=r, attack=atk["size"], repair=rep["size"])); break
        def_cost += rep["size"]; fixed = merge_iv(fixed, rep["intervention"])
        rounds.append(dict(round=r, attack=atk["size"], strategy=atk["strategy"], repair=rep["size"], move=rep["move"], fragile=rep.get("fragile")))
        print(f"    {model} round {r}: attack {atk['size']} ({atk['strategy'].replace('attacker-', '')}) -> repair {rep['size']} ({rep['move']}{', FRAGILE' if rep.get('fragile') else ''})   cum {atk_cost}/{def_cost}", flush=True)
    if winner is None: winner = f"undecided after {args.rounds} rounds"
    RULES["moves"]["inject"]["max_hz"] = DEFAULT_HZ; DEFENCE_RULES["readout_bias_mv"] = 0.0
    return dict(model=model, single_shot=base["size"] if base else None, single_shot_strategy=base["strategy"] if base else None,
                winner=winner, attacker_cost=atk_cost, defender_cost=def_cost,
                exchange_ratio=round(def_cost / atk_cost, 2) if atk_cost else None, rounds=rounds)


desc = {"T0": "current rules", "T1": f"inject capped at {args.cap_hz:.0f} Hz", "T2": "attacker cannot touch readout's direct inputs",
        "T3": f"defender may bias the readout +{args.readout_mv:.0f} mV"}
print(f"{args.sex} {b.name}: iterated game under threat models (budget {args.budget} each, robust defender)\n")
results = []
for m in args.models:
    print(f"  [{m}] {desc[m]}", flush=True)
    results.append(play(m))
print(f"\n{'model':6s} {'rules':46s} {'1-shot attack':>13s} {'winner':38s} {'atk':>4s} {'def':>4s} {'ratio':>6s} rounds")
for r in results:
    print(f"{r['model']:6s} {desc[r['model']]:46s} {str(r['single_shot']):>13s} {r['winner']:38s} {r['attacker_cost']:>4d} {r['defender_cost']:>4d} "
          f"{str(r['exchange_ratio']):>6s} {len(r['rounds'])}")
os.makedirs(f"atlas/{args.sex}", exist_ok=True)
json.dump(dict(sex=args.sex, behaviour=b.name, budget=args.budget, models=results, seconds=round(time.time() - t0)),
          open(f"atlas/{args.sex}/threatmodels_{b.name}.json", "w"), indent=1, default=str)
print(f"saved atlas/{args.sex}/threatmodels_{b.name}.json  ({time.time()-t0:.0f}s)")
