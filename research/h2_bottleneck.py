"""H2 (revised, pre-registration section 7): how much does an attack cost when every direct input of the output is
forbidden? Same three strategies as the baseline (greedy, inject inhibitors, weaken only); the agents fall back one step
upstream when all direct candidates are excluded. Best attack checked on the 15 unseen seeds.

Usage: python research/h2_bottleneck.py <instance> <goal>        output: research/results/h2/<instance>_<goal>.json
"""
import json, os, sys, time
import numpy as np
from guard import start, finish
out = start("h2")
import instances
from behaviours import Behaviour
from agents import Attacker, Log

NAME, GOAL = sys.argv[1], sys.argv[2]; FRESH = tuple(range(11, 26))
SC = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else None
if NAME.startswith("s_"):                     # screen pathway (27 Sep): that fly's primary model, or --scale
    FLY = NAME.split("_")[2]; brain = instances.screen_brain(FLY, SC); instances.register_screen(brain, FLY)
elif NAME.startswith("b_"):
    from banc import BancBrain
    brain = BancBrain(scale=json.load(open(os.path.join(os.path.dirname(out), "banc", "scale_factor.json")))["median_ratio"]); instances.register_banc()
else:
    from brain import Brain
    brain = Brain("female"); instances.register(brain=brain)
t0 = time.time(); b = Behaviour(brain, NAME); log = Log(quiet=True)
W = brain.W.tocsc(); direct = sorted({int(i) for r in b.readouts for i in W[:, r].indices} - set(b.readouts))
att = [Attacker(b, log, GOAL, excluded=direct) for _ in range(3)]
found = [a for a in (att[0].greedy(), att[1].inject_inhibitors(), att[2].weaken_only()) if a]
row = dict(instance=NAME, goal=GOAL, n_direct_inputs_forbidden=len(direct),
           attacks_found=[dict(strategy=a["strategy"], size=a["size"]) for a in found])
if found:
    best = min(found, key=lambda a: (a["size"], a["simulations"])); iv = best["intervention"]
    r = [b.measure(GOAL in ("block", "corrupt"), seed=s, **iv)[0] for s in FRESH]
    ok = [x <= b.off_thr for x in r] if GOAL == "block" else [x >= b.on_thr for x in r]
    row.update(best_strategy=best["strategy"], best_size=best["size"], goal_met=int(sum(ok)), found=bool(sum(ok) >= 12),
               intervention={k: (list(map(int, v)) if isinstance(v, list) else {str(i): float(x) for i, x in v.items()}) for k, v in iv.items()})
else:
    row.update(best_size=None, found=False)
row["seconds"] = round(time.time() - t0)
row["scale"] = getattr(brain, "scale", 1.0)
json.dump(row, open(os.path.join(out, f"{NAME}_{GOAL}{'_x%s' % SC if SC else ''}.json"), "w"), indent=1)
print(f"{NAME} {GOAL}: {len(direct)} direct inputs forbidden -> "
      + (f"attack {row['best_size']} via {row['best_strategy']} ({row['goal_met']}/15)" if found else "NO attack within the rules") + f" | {row['seconds']}s", flush=True)
finish()
