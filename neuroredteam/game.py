"""Deeper red/blue team: collateral damage, stealth attacks, and the iterated attack <-> repair game.

collateral(iv)        : which OTHER behaviours (same sex) an intervention breaks (two-condition test on 3 seeds).
stealth               : among all minimal block attacks found (chokepoints-style enumeration), the smallest one with
                        zero collateral; reports collateral of every attack.
iterated game         : round r: attacker attacks the brain carrying all earlier attacks + repairs -> defender repairs
                        (with everything fixed) -> next round. Ends when one side fails within budget or after --rounds.
                        Reports cumulative attacker / defender cost and who won. The 'game value' of a behaviour.
Usage: python game.py --sex female --behaviour eat --mode stealth|game|both [--rounds 5]
"""
import argparse, json, os, time
import numpy as np
from brain import Brain
from behaviours import Behaviour, BEHAVIOURS, RULES
from agents import Attacker, Defender, Log, merge_iv, size_of, CHECK_SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--behaviour", default="eat")
ap.add_argument("--mode", default="both", choices=["stealth", "game", "both"]); ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--search", action="store_true", help="use beam search in the game rounds too (slower)")
ap.add_argument("--defender", default="robust", choices=["greedy", "robust"])
ap.add_argument("--budget", type=int, default=12, help="cumulative neurons each side may touch over the whole game")
args = ap.parse_args()

brain = Brain(args.sex); b = Behaviour(brain, args.behaviour); log = Log(quiet=True); t0 = time.time()
others = {n: Behaviour(brain, n) for n, c in BEHAVIOURS[args.sex].items()
          if n != args.behaviour and c.get("expected", "on") == "on" and c.get("batch", True)}
print(f"{args.sex} {b.name}; collateral measured on: {list(others)}")


def collateral(iv, seeds=CHECK_SEEDS):
    """Other behaviours broken by iv: {name: (on_hz, off_hz)} for those failing is_normal on any seed."""
    broken = {}
    for n, ob in others.items():
        for s in seeds:
            ok, on, off = ob.is_normal(seed=s, **iv)
            if not ok: broken[n] = (round(on, 1), round(off, 1)); break
    return broken


def neurons_of(iv):
    return frozenset(size_set(iv))


def size_set(iv):
    return set(iv.get("lesion", [])) | set(iv.get("inject", {})) | set(iv.get("weaken", {}))


result = {"sex": args.sex, "behaviour": b.name}

# ---------------------------------------------------------------- stealth
if args.mode in ("stealth", "both"):
    goal = "corrupt" if b.expected == "off" else "block"
    strats = ("greedy",) if goal == "corrupt" else ("greedy", "inject_inhibitors", "weaken_only")
    found = {}
    def add(atk):
        if atk: found.setdefault(neurons_of(atk["intervention"]), atk)
    for st in strats: add(getattr(Attacker(b, log, goal), st)())
    seen = set()
    for _ in range(2):
        for S in list(found):
            for n in S:
                if n in seen: continue
                seen.add(n)
                for st in strats: add(getattr(Attacker(b, log, goal, excluded={n}), st)())
    keys = [k for k in found if not any(o < k for o in found)]
    rows = []
    for k in sorted(keys, key=len):
        atk = found[k]; col = collateral(atk["intervention"])
        rows.append(dict(size=len(k), strategy=atk["strategy"], neurons=sorted(k), types=[brain.type_of(i) for i in sorted(k)],
                         collateral=col))
        print(f"  attack size {len(k)} ({atk['strategy']:22s}) {[brain.type_of(i) for i in sorted(k)]}  collateral: {col or 'none'}")
    stealthy = [r for r in rows if not r["collateral"]]
    best = min(stealthy, key=lambda r: r["size"]) if stealthy else None
    print(f"STEALTH: {len(rows)} minimal attacks, {len(stealthy)} with zero collateral; smallest stealthy = "
          f"{best['size'] if best else 'none'}  (smallest overall = {min(r['size'] for r in rows) if rows else 'none'})")
    result["stealth"] = dict(attacks=rows, smallest=min((r["size"] for r in rows), default=None),
                             smallest_stealthy=best["size"] if best else None)

# ---------------------------------------------------------------- iterated game
if args.mode in ("game", "both") and b.expected == "on":
    fixed = {}; rounds = []; atk_cost = def_cost = 0; winner = None
    for r in range(1, args.rounds + 1):
        A = Attacker(b, log, "block", fixed=fixed)
        cands = [a for a in (A.greedy(name=f"r{r}-greedy"), Attacker(b, log, "block", fixed=fixed).inject_inhibitors(name=f"r{r}-inject"),
                             Attacker(b, log, "block", fixed=fixed).weaken_only(name=f"r{r}-weaken")) if a]
        if args.search:
            s_ = Attacker(b, log, "block", fixed=fixed).search(name=f"r{r}-search", max_size=max(1, min(a["size"] for a in cands) - 1) if cands else 4)
            if s_: cands.append(s_)
        atk = min(cands, key=lambda a: (a["size"], a["simulations"])) if cands else None
        if not atk:
            winner = "defender"; rounds.append(dict(round=r, attack=None)); print(f"  round {r}: attacker finds nothing within budget -> DEFENDER wins"); break
        if atk_cost + atk["size"] > args.budget:
            winner = "defender"; rounds.append(dict(round=r, attack=atk["size"], note="attacker out of budget"))
            print(f"  round {r}: attacker needs {atk['size']} more but has {args.budget - atk_cost} left -> DEFENDER wins (attacker out of budget)"); break
        atk_cost += atk["size"]; fixed = merge_iv(fixed, atk["intervention"])
        col = collateral(fixed)
        D = Defender(b, {"intervention": fixed}, log)
        rep = D.robust(name=f"r{r}-defender") if args.defender == "robust" else D.greedy(name=f"r{r}-defender")
        if not rep:
            winner = "attacker"; rounds.append(dict(round=r, attack=atk["size"], attack_types=[brain.type_of(i) for i in size_set(atk["intervention"])], repair=None, collateral=col))
            print(f"  round {r}: attack {atk['size']} ({atk['strategy']}); defender finds no repair -> ATTACKER wins   (cumulative attack {atk_cost}, defence {def_cost})"); break
        if def_cost + rep["size"] > args.budget:
            winner = "attacker"; rounds.append(dict(round=r, attack=atk["size"], repair=rep["size"], note="defender out of budget"))
            print(f"  round {r}: attack {atk['size']}; repair would cost {rep['size']} but defender has {args.budget - def_cost} left -> ATTACKER wins (defender out of budget)"); break
        def_cost += rep["size"]; fixed = merge_iv(fixed, rep["intervention"])
        rounds.append(dict(round=r, attack=atk["size"], attack_strategy=atk["strategy"], attack_types=[brain.type_of(i) for i in size_set(atk["intervention"])],
                           repair=rep["size"], repair_move=rep["move"], on_hz=rep["on_hz"], collateral=col))
        tag = "" if "fragile" not in rep else (", FRAGILE" if rep["fragile"] else ", probe-robust")
        print(f"  round {r}: attack {atk['size']} ({atk['strategy'].replace('attacker-', '')}) -> repair {rep['size']} ({rep['move']}, {rep['on_hz']:.0f} Hz{tag})"
              f"   cumulative attack {atk_cost} / defence {def_cost}   collateral {col or 'none'}", flush=True)
    if winner is None: winner = f"undecided after {args.rounds} rounds"
    ratio = (def_cost / atk_cost) if atk_cost else float("nan")
    print(f"GAME: {winner}; cumulative attacker cost {atk_cost}, defender cost {def_cost}  (exchange ratio defence/attack = {ratio:.2f}, budget {args.budget} each)")
    result["game"] = dict(winner=winner, rounds=rounds, attacker_cost=atk_cost, defender_cost=def_cost, exchange_ratio=ratio, budget=args.budget)

result["seconds"] = round(time.time() - t0)
os.makedirs(f"atlas/{args.sex}", exist_ok=True)
result["defender"] = args.defender
json.dump(result, open(f"atlas/{args.sex}/game_{b.name}_{args.defender}.json", "w"), indent=1, default=str)
print(f"saved atlas/{args.sex}/game_{b.name}_{args.defender}.json  ({result['seconds']}s)")
