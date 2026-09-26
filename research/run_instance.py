"""Red team + blue team + independent verification for ONE research instance, exactly as Phase 3 verified its cases
(neuroredteam/verify_phase3.py, defender 'restore', target 80 %): the same three attack strategies (greedy,
inject inhibitors, weaken only), the same repair (restore / restore_off with the control behaviour), and every result
re-checked on 15 unseen seeds (11-25), judged seed by seed: SOLID 15/15, MARGINAL 12-14, FAIL below 12.

Output: research/results/instances/<name>_<goal>.json   (never neuroredteam/atlas/)
Usage:  python research/run_instance.py r_escape_R block
"""
import json, os, sys, time
import numpy as np
from guard import start, finish
out_dir = start(sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else "instances")
import instances
from brain import Brain
from behaviours import Behaviour, BEHAVIOURS
from agents import Attacker, Defender, Log, merge_iv

NAME, GOAL = sys.argv[1], sys.argv[2]
FRESH = tuple(range(11, 26)); TARGET = 0.8
if NAME.startswith("s_"):                     # screen pathway: s_<k>_<fly>, that fly's primary (pre-registered) model
    FLY = NAME.split("_")[2]; SCD = os.path.join(os.path.dirname(out_dir), "{}", "scale_factor.json")
    if FLY == "BANC":
        from banc import BancBrain; brain = BancBrain(scale=json.load(open(SCD.format("banc")))["median_ratio"])
    elif FLY == "male":
        from malecns import MaleBrain; brain = MaleBrain(scale=json.load(open(SCD.format("male")))["median_ratio"])
    else:
        brain = Brain("female")
    instances.register_screen(brain, FLY)
elif NAME.startswith("b_"):                     # BANC: the pre-registered scaled weights (research/banc.py)
    from banc import BancBrain
    UNSCALED = "--unscaled" in sys.argv                  # the pre-registered secondary BANC model (weights x 1)
    brain = BancBrain(scale=1.0 if UNSCALED else json.load(open(os.path.join(os.path.dirname(out_dir), "banc", "scale_factor.json")))["median_ratio"])
    instances.register_banc()
    if NAME == "b_bitter_R": GOAL_NOTE = "attack only counts (control excluded)"
elif NAME.startswith("m_"):                   # MaleCNS: traced neurons only, the pre-registered scaled weights
    from malecns import MaleBrain
    brain = MaleBrain(scale=1.0 if "--unscaled" in sys.argv else json.load(open(os.path.join(os.path.dirname(out_dir), "male", "scale_factor.json")))["median_ratio"])
    instances.register_male()
else:
    brain = Brain("female"); instances.register(brain=brain)


def rates(b, with_stim, iv): return [b.measure(with_stim, seed=s, **iv)[0] for s in FRESH]


def grade(passes):
    n = sum(passes); return ("SOLID" if n == len(passes) else "MARGINAL" if n >= len(passes) - 3 else "FAIL"), n


t0 = time.time(); b = Behaviour(brain, NAME); log = Log(quiet=True); cfg = BEHAVIOURS[brain.sex][NAME]
row = dict(instance=NAME, role=(instances.role(NAME) if NAME in instances.INSTANCES else "confirm" if NAME.startswith("s_") else "confirm (BANC)"), goal=GOAL, fresh_seeds=list(FRESH),
           dataset=getattr(brain, "dataset", "FlyWire v783"))
healthy_on = float(np.mean(rates(b, True, {}))); row["healthy_on"] = round(healthy_on, 1)
row["healthy_off"] = round(float(np.mean(rates(b, False, {}))), 1)

# --- 1. the attack (same strategies as verify_phase3.py without --quick) -----------------------------------------
strategies = [Attacker(b, log, GOAL).greedy(), Attacker(b, log, GOAL).inject_inhibitors(), Attacker(b, log, GOAL).weaken_only()]
found = [a for a in strategies if a]
row["attacks_found"] = [dict(strategy=a["strategy"], size=a["size"], intervention=a["intervention"]) for a in found]
atk = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
if atk:
    iv = atk["intervention"]; row["attack"] = dict(strategy=atk["strategy"], size=atk["size"], intervention=iv)
    ra = rates(b, GOAL in ("block", "corrupt"), iv)
    ok = [r <= b.off_thr for r in ra] if GOAL == "block" else [r >= b.on_thr for r in ra]
    g, n = grade(ok); row.update(attack_grade=g, attack_seeds_ok=n, after_attack=ra)
    # --- 2. the repair ---------------------------------------------------------------------------------------------
    D = Defender(b, {"intervention": iv}, log)
    ctrl_name = cfg.get("control"); ctrl = Behaviour(brain, ctrl_name) if ctrl_name else None
    rep = D.restore_off(control=ctrl) if GOAL == "corrupt" else D.restore(target=TARGET)
    if rep and "intervention" in rep:
        full = merge_iv(iv, rep["intervention"])
        r_on, r_off = rates(b, True, full), rates(b, False, full)
        ok = [x <= b.off_thr for x in r_on] if b.expected == "off" else [x >= b.on_thr for x in r_on]
        g, n = grade(ok)
        row["repair"] = dict(size=rep["size"], move=rep.get("move"), intervention=rep["intervention"],
                             partial=bool(rep.get("partial")), below_target=bool(rep.get("below_target")), why=rep.get("why"))
        row.update(repair_grade=g, repair_seeds_ok=n, after_repair_on=r_on, after_repair_off=r_off,
                   cheats=any(x > b.off_thr for x in r_off))
        if b.expected != "off" and healthy_on > 0:
            row["recovery_worst"] = round(min(r_on) / healthy_on, 3); row["recovery_mean"] = round(float(np.mean(r_on)) / healthy_on, 3)
        if ctrl:
            rc = rates(ctrl, True, full); row["control"] = dict(name=ctrl_name, hz=rc, ok=all(x >= ctrl.on_thr for x in rc))
    else:
        row["repair"] = None
else:
    row["attack"] = None
row["seconds"] = round(time.time() - t0)


def plain(v):
    """JSON-safe copy: tuple keys (e.g. potentiate {(pre, post): factor}) become "pre,post" strings; numpy -> Python"""
    if isinstance(v, dict): return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    return v.item() if hasattr(v, "item") else v


suffix = "_unscaled" if "--unscaled" in sys.argv else ""; row["scale"] = getattr(brain, "scale", 1.0)
json.dump(plain(row), open(os.path.join(out_dir, f"{NAME}_{GOAL}{suffix}.json"), "w"), indent=1)
a_ = row.get("attack") or {}; r_ = row.get("repair") or {}
print(f"{NAME} {GOAL}: attack {a_.get('size')} ({row.get('attack_grade')} {row.get('attack_seeds_ok')}/15) | "
      f"repair {r_.get('size')} {'(PARTIAL: blue team loses)' if r_.get('partial') else ''} ({row.get('repair_grade')} {row.get('repair_seeds_ok')}/15)"
      f"{' recovery %.0f-%.0f%%' % (100 * row['recovery_worst'], 100 * row['recovery_mean']) if 'recovery_worst' in row else ''}"
      f"{' control ' + ('OK' if row['control']['ok'] else 'BROKEN') if 'control' in row else ''} | {row['seconds']}s", flush=True)
finish()
