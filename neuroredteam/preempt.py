"""Pre-emptive defence (the head start): the defender hardens the HEALTHY brain with k neurons before any attack, the
behaviour must still pass the two-condition test, then the attacker attacks the hardened brain.

Hardening plans (defender moves only; the readout itself is never touched):
  depress   : weaken the strongest inhibitory synapses onto the readout (x min_factor) -- disarms 'inject inhibitors'
  potentiate: strengthen the strongest excitatory synapses onto the readout
  bias      : hold the strongest drivers closer to threshold
  mixed     : depress the top half, potentiate the rest
For each k and plan: check is_normal on all seeds, run the attacker (greedy, inject_inhibitors, weaken_only, and
search with --search), record the minimal attack size. The plan with the largest minimal attack wins at that k.
Then, for the best k (--game), the iterated game is played from the hardened state with the defender's remaining budget.
Usage: python preempt.py --sex female --behaviour eat [--ks 2 4 6 8 12] [--search] [--game]
"""
import argparse, json, os, time
import numpy as np
from brain import Brain
from behaviours import Behaviour, RULES, DEFENCE_RULES
from agents import Attacker, Defender, Log, merge_iv, size_of, CHECK_SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8, 12]); ap.add_argument("--search", action="store_true")
ap.add_argument("--game", action="store_true", help="play the iterated game from the best hardened state")
ap.add_argument("--budget", type=int, default=12)
args = ap.parse_args()

brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); log = Log(quiet=True); t0 = time.time()
ro = b.readout; W = brain.W
dep = DEFENCE_RULES["moves"]["depress"]["min_factor"]; maxf = DEFENCE_RULES["moves"]["potentiate"]["max_factor"]
mv = DEFENCE_RULES["moves"]["bias"]["max_mv"]

# candidate pools, ranked by what the ATTACKER likes: its inhibitor list (silent GABA inputs it would inject into),
# and its driver list (the neurons it would cut)
A0 = Attacker(b, log, "block")
inhib = A0.inhibitors()                     # strongest inhibitory inputs to the readout
drivers = A0.drivers()                      # active excitatory drivers, strongest first
plans = {
    "depress":    lambda k: {"potentiate": {(i, ro): dep for i in inhib[:k]}},
    "potentiate": lambda k: {"potentiate": {(i, ro): maxf for i in drivers[:k]}},
    "bias":       lambda k: {"bias": {i: mv for i in drivers[:k]}},
    "mixed":      lambda k: {"potentiate": {**{(i, ro): dep for i in inhib[:k // 2]}, **{(i, ro): maxf for i in drivers[:k - k // 2]}}},
}


def attack_size(fixed, tag):
    """Smallest block attack on a brain carrying `fixed` (all strategies, + search if asked). None = not found."""
    found = [a for a in (Attacker(b, log, "block", fixed=fixed).greedy(name=f"{tag}-greedy"),
                         Attacker(b, log, "block", fixed=fixed).inject_inhibitors(name=f"{tag}-inject"),
                         Attacker(b, log, "block", fixed=fixed).weaken_only(name=f"{tag}-weaken")) if a]
    if args.search:
        cap = (min(a["size"] for a in found) - 1) if found else RULES["moves"]["lesion"]["max"]
        if cap >= 1:
            s_ = Attacker(b, log, "block", fixed=fixed).search(name=f"{tag}-search", max_size=cap)
            if s_: found.append(s_)
    best = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
    return best


def normal(iv):
    return all(b.is_normal(seed=s, **iv)[0] for s in CHECK_SEEDS)


base = attack_size({}, "base")
print(f"{args.sex} {b.name}: baseline minimal block attack = {base['size'] if base else 'none'} ({base['strategy'] if base else ''})\n")
print(f"{'k':>3s} | " + " | ".join(f"{p:>12s}" for p in plans) + " |  best plan -> attack size")
rows = []; best_state = None
for k in args.ks:
    line = f"{k:>3d} | "; res = {}
    for name, make in plans.items():
        iv = make(k); iv = {kk: v for kk, v in iv.items() if v}
        if size_of(iv) == 0: res[name] = ("n/a", None); line += f"{'n/a':>12s} | "; continue
        if not normal(iv):
            on = np.mean([b.measure(True, seed=s, **iv)[0] for s in CHECK_SEEDS]); off = np.mean([b.measure(False, seed=s, **iv)[0] for s in CHECK_SEEDS])
            res[name] = (f"breaks fly ({on:.0f}/{off:.0f})", None); line += f"{'breaks':>12s} | "; continue
        atk = attack_size(iv, f"k{k}-{name}")
        sz = atk["size"] if atk else None
        res[name] = (sz if sz else "unbreakable", iv); line += f"{str(sz) if sz else 'UNBREAKABLE':>12s} | "
    scored = [(999 if r[0] == "unbreakable" else r[0], name, r[1]) for name, r in res.items() if r[1] is not None and r[0] != "n/a"]
    if scored:
        s, name, iv = max(scored)
        line += f"  {name} -> {'unbreakable' if s == 999 else s}"
        rows.append(dict(k=k, plan=name, attack_size=None if s == 999 else s, unbreakable=s == 999,
                         all={n: (r[0] if not isinstance(r[0], str) else r[0]) for n, r in res.items()}))
        if best_state is None or s > best_state[0]: best_state = (s, k, name, iv)
    print(line, flush=True)

result = dict(sex=args.sex, behaviour=b.name, baseline=base["size"] if base else None, sweep=rows)
if best_state:
    s, k, name, iv = best_state
    print(f"\nHEAD START: investing {k} neurons in '{name}' raises the minimal attack from {base['size'] if base else 'none'} to "
          f"{'UNBREAKABLE' if s == 999 else s}  (defender cost {k}, attacker cost {'inf' if s == 999 else s})")
    result["best"] = dict(k=k, plan=name, attack_size=None if s == 999 else s,
                          intervention={kk: ({str(a): c for a, c in v.items()} if isinstance(v, dict) else v) for kk, v in iv.items()})

    if args.game and s != 999:
        print(f"\nITERATED GAME from the hardened brain (defender already spent {k} of {args.budget}):")
        fixed = dict(iv); atk_cost, def_cost, winner, rounds = 0, k, None, []
        for r in range(1, 9):
            atk = attack_size(fixed, f"g{r}")
            if not atk: winner = "defender"; print(f"  round {r}: attacker finds nothing -> DEFENDER wins"); break
            if atk_cost + atk["size"] > args.budget: winner = "defender"; print(f"  round {r}: attacker out of budget -> DEFENDER wins"); break
            atk_cost += atk["size"]; fixed = merge_iv(fixed, atk["intervention"])
            rep = Defender(b, {"intervention": fixed}, log).robust(name=f"g{r}-defender")
            if not rep: winner = "attacker"; print(f"  round {r}: attack {atk['size']}; no repair -> ATTACKER wins"); break
            if def_cost + rep["size"] > args.budget:
                winner = "attacker"; print(f"  round {r}: attack {atk['size']}; repair {rep['size']} exceeds budget -> ATTACKER wins"); break
            def_cost += rep["size"]; fixed = merge_iv(fixed, rep["intervention"])
            rounds.append(dict(round=r, attack=atk["size"], repair=rep["size"], move=rep["move"], fragile=rep.get("fragile")))
            print(f"  round {r}: attack {atk['size']} ({atk['strategy'].replace('attacker-', '')}) -> repair {rep['size']} ({rep['move']}{', FRAGILE' if rep.get('fragile') else ''})"
                  f"   cumulative attack {atk_cost} / defence {def_cost}", flush=True)
        if winner is None: winner = "undecided"
        print(f"GAME with head start: {winner}; attacker {atk_cost}, defender {def_cost} (exchange ratio {def_cost / max(atk_cost, 1):.2f})")
        result["game"] = dict(winner=winner, attacker_cost=atk_cost, defender_cost=def_cost, rounds=rounds)

os.makedirs(f"atlas/{args.sex}", exist_ok=True)
json.dump(result, open(f"atlas/{args.sex}/preempt_{b.name}.json", "w"), indent=1, default=str)
print(f"saved atlas/{args.sex}/preempt_{b.name}.json  ({time.time() - t0:.0f}s)")
