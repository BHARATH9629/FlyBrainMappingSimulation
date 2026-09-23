"""Phase 3 verification: every behaviour, every goal, attack AND repair, checked independently.

The agents screen candidates on seeds 1-3 and accept them only if they hold on seeds 1-8. This script re-checks every
result on 15 FRESH seeds (11-25) never used by either agent, and judges EVERY SEED SEPARATELY: a mean can look fine
while individual runs cross the bar, which is exactly how a marginal attack hides.
Verdict per result: SOLID = holds on 15/15 seeds, MARGINAL = 12-14/15, FAIL = fewer.

For each (behaviour, goal) it asks:
  1. did the attacker find an attack, and does it really meet the goal on fresh seeds?
  2. did the defender find a repair, and does it really restore the behaviour on fresh seeds?
  3. does the repair cheat?  (a repaired behaviour must still be silent with no stimulus)
  4. for a corrupt repair: does the control behaviour still work (no fixing bitter by breaking feeding)?
  5. FALSE PASS: does an agent report success where the independent check disagrees?  This is the dangerous case.

Usage: python verify_phase3.py [--sex female] [--behaviour eat] [--quick]
"""
import argparse, json, os, sys, time
import numpy as np
from brain import Brain
from behaviours import Behaviour, BEHAVIOURS
from agents import Attacker, Defender, Log, merge_iv

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"])
ap.add_argument("--behaviour", default=None, help="just one behaviour (default: all)")
ap.add_argument("--quick", action="store_true", help="skip the slow extra attack strategies")
ap.add_argument("--defender", default="restore", choices=["restore", "greedy"],
                help="restore = repair to >= 80%% of healthy (default); greedy = the old 'just clear the bar' defender")
ap.add_argument("--target", type=float, default=0.8, help="restore: fraction of healthy that every seed must reach")
args = ap.parse_args()
FRESH = tuple(range(11, 26))               # 15 seeds never used by the agents (they search on 1-3, accept on 1-8)
CASES = [("eat", "block"), ("eat", "force"), ("drink", "block"), ("escape", "block"),
         ("retreat", "block"), ("turn", "block"), ("refuse_bitter", "corrupt")]
if args.behaviour: CASES = [c for c in CASES if c[0] == args.behaviour]

brain = Brain(args.sex)
rows = []; problems = []


def rates(b, with_stim, iv, seeds=FRESH):
    return [b.measure(with_stim, seed=s, **iv)[0] for s in seeds]


def mean_hz(b, with_stim, iv, seeds=FRESH):
    return float(np.mean(rates(b, with_stim, iv, seeds)))


def grade(passes):
    """per-seed pass count -> SOLID / MARGINAL / FAIL"""
    n = sum(passes); return "SOLID" if n == len(passes) else ("MARGINAL" if n >= len(passes) - 3 else "FAIL"), n


def check(name, goal):
    b = Behaviour(brain, name); log = Log(quiet=True)
    cfg = BEHAVIOURS[args.sex][name]
    row = {"behaviour": name, "goal": goal}
    healthy_on = mean_hz(b, True, {}); healthy_off = mean_hz(b, False, {})
    row["healthy_on"], row["healthy_off"] = round(healthy_on, 1), round(healthy_off, 1)

    # --- 1. the attack ------------------------------------------------------
    strategies = [Attacker(b, log, goal).greedy(), Attacker(b, log, goal).inject_inhibitors()]
    if not args.quick: strategies.append(Attacker(b, log, goal).weaken_only())
    found = [a for a in strategies if a]
    atk = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
    if not atk:
        row["attack"] = None; problems.append(f"{name}/{goal}: NO ATTACK FOUND"); rows.append(row); return row
    iv = atk["intervention"]
    row["attack"] = f"{atk['size']} via {atk['strategy'].replace('attacker-', '')}"
    ra = rates(b, goal in ("block", "corrupt"), iv)
    after = float(np.mean(ra)); row["after_attack_hz"] = round(after, 1)
    ok_seed = [r <= b.off_thr for r in ra] if goal == "block" else [r >= b.on_thr for r in ra]   # force/corrupt want HIGH
    g, n = grade(ok_seed); row["attack_grade"] = g; row["attack_seeds_ok"] = f"{n}/{len(ra)}"
    row["attack_worst_hz"] = round(max(ra) if goal == "block" else min(ra), 1)
    atk_ok = g == "SOLID"; row["attack_verified"] = atk_ok
    if not atk_ok: problems.append(f"{name}/{goal}: attack {g} -- holds on {n}/{len(ra)} unseen seeds "
                                   f"(mean {after:.1f} Hz, worst {row['attack_worst_hz']} Hz)")

    # --- 2. the repair ------------------------------------------------------
    D = Defender(b, {"intervention": iv}, log)
    ctrl_name = cfg.get("control")
    ctrl = Behaviour(brain, ctrl_name) if ctrl_name else None
    # block and force both repair with restore(): its test (back to the target level WITH the stimulus, silent WITHOUT)
    # covers what a force attack breaks, and the live window allows F then D, so that path is tested like any other.
    if goal == "corrupt": rep = D.restore_off(control=ctrl)
    elif args.defender == "restore":
        rep = D.restore(target=args.target)           # below target: restore() returns its own strongest working repair
        if rep.get("below_target"): row["below_target"] = rep.get("why")
    else: rep = D.greedy()
    if not rep or "intervention" not in rep:
        row["repair"] = None; problems.append(f"{name}/{goal}: NO REPAIR FOUND"); rows.append(row); return row
    full = merge_iv(iv, rep["intervention"])
    row["repair"] = f"{rep['size']} via {rep['move']}"
    r_on = rates(b, True, full); r_off = rates(b, False, full)
    on, off = float(np.mean(r_on)), float(np.mean(r_off))
    row["after_repair_on"], row["after_repair_off"] = round(on, 1), round(off, 1)
    if b.expected != "off" and healthy_on > 0:           # how much of the healthy behaviour came back, worst held-out run
        row["recovery_worst"] = round(min(r_on) / healthy_on, 3); row["recovery_mean"] = round(on / healthy_on, 3)
    if rep.get("curve"): row["cost_recovery_curve"] = rep["curve"]

    if b.expected == "off":                               # refuse_bitter: the repair must SILENCE the readout again
        ok_seed = [x <= b.off_thr for x in r_on]
        row["repair_target"] = f"<= {b.off_thr:.0f} Hz with the refused stimulus"
    else:
        ok_seed = [x >= b.on_thr for x in r_on]
        row["repair_target"] = f">= {b.on_thr:.0f} Hz with the stimulus"
    g, n = grade(ok_seed); row["repair_grade"] = g; row["repair_seeds_ok"] = f"{n}/{len(r_on)}"
    rep_ok = g == "SOLID"
    cheat = any(x > b.off_thr for x in r_off)             # fires with NO stimulus on ANY seed: not a repair
    row["repair_verified"] = rep_ok and not cheat
    row["no_forcing_ok"] = not cheat
    if rep.get("partial"): problems.append(f"{name}/{goal}: BLUE TEAM LOST -- {rep.get('why', 'partial repair only')}")
    elif not rep_ok: problems.append(f"{name}/{goal}: repair {g} -- restores on {n}/{len(r_on)} unseen seeds (mean {on:.1f} Hz, "
                                     f"needs {'<=' if b.expected == 'off' else '>='} {b.off_thr if b.expected == 'off' else b.on_thr:.0f})")
    if cheat: problems.append(f"{name}/{goal}: CHEATING REPAIR -- fires up to {max(r_off):.1f} Hz with no stimulus")
    if ctrl:
        rc = rates(ctrl, True, full); c = float(np.mean(rc))
        row["control_hz"] = round(c, 1); row["control_ok"] = all(x >= ctrl.on_thr for x in rc)
        if not row["control_ok"]: problems.append(f"{name}/{goal}: repair broke the control behaviour '{ctrl_name}' ({c:.1f} Hz)")
    rows.append(row); return row


t0 = time.time()
print(f"Phase 3 verification, {args.sex} fly -- every result re-checked on unseen seeds {FRESH}\n", flush=True)
for name, goal in CASES:
    r = check(name, goal)
    verdict = "OK" if r.get("attack_verified") and r.get("repair_verified") else "PROBLEM"
    rec = f" ({r['recovery_worst'] * 100:.0f}-{r['recovery_mean'] * 100:.0f}%)" if "recovery_worst" in r else ""
    print(f"  {name:14s} {goal:8s} attack {str(r.get('attack')):20s} {str(r.get('attack_seeds_ok')):6s} {r.get('after_attack_hz')} Hz | "
          f"repair {str(r.get('repair'))[:30]:30s} {str(r.get('repair_seeds_ok')):6s} {r.get('after_repair_on')} Hz{rec}  [{verdict}]", flush=True)

print(f"\n{'-' * 100}")
if problems:
    print(f"{len(problems)} PROBLEM(S):")
    for p in problems: print(f"  ! {p}")
else:
    print("all cases verified on unseen seeds: attacks meet their goal, repairs restore the behaviour, none cheats")
print(f"({time.time() - t0:.0f}s)")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas", args.sex, f"verify_phase3_{args.defender}.json")
json.dump({"sex": args.sex, "fresh_seeds": list(FRESH), "cases": rows, "problems": problems}, open(out, "w"), indent=1)
print(f"saved atlas/{args.sex}/verify_phase3_{args.defender}.json")
