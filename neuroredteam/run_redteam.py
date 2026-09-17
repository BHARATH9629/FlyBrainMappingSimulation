"""NeuroRedTeam, milestone 1: automatic attack and repair of the 'eat' behaviour.
Usage:  python run_redteam.py [--behaviour eat] [--attacker greedy|random]"""
import argparse, json, time, os
import numpy as np
from brain import Brain
from behaviours import Behaviour
from agents import Attacker, Defender, Log

ap = argparse.ArgumentParser()
ap.add_argument("--behaviour", default="eat"); ap.add_argument("--attacker", default="greedy", choices=["greedy", "random"])
args = ap.parse_args()

t0 = time.time()
brain = Brain(); print(f"brain loaded: {brain.N:,} neurons, {brain.W.nnz:,} synapses  [{time.time()-t0:.0f}s]")
b = Behaviour(brain, args.behaviour)

print(f"\n== Healthy baseline for '{b.name}' ==")
t1 = time.time(); ok, on, off = b.is_normal()
print(f"  readout with stimulus {on:.1f} Hz, without {off:.1f} Hz -> {'normal' if ok else 'NOT normal (check thresholds)'}"
      f"   (sub-network {b.measure(True)[1]['n_sub']:,} neurons, {time.time()-t1:.1f}s for two runs)")

print(f"\n== RED TEAM: block '{b.name}' with the fewest allowed changes ==")
log = Log(); atk = Attacker(b, log)
attack = atk.greedy() if args.attacker == "greedy" else atk.random()
if attack is None:
    print("  attacker failed within budget"); raise SystemExit
print(f"  -> ATTACK FOUND: {attack['move']} {attack['size']} neurons {attack['neurons']}  ({attack['simulations']} simulations)")

print(f"\n== BLUE TEAM: repair '{b.name}' without undoing the attack ==")
dfn = Defender(b, attack, log); repair = dfn.greedy()
if repair is None:
    print("  defender failed within budget (this is itself an atlas entry: hard to repair)")
else:
    print(f"  -> REPAIR FOUND: {repair['move']} on {repair['size']} neurons {repair['neurons']}; "
          f"readout {repair['on_hz']:.1f} Hz with stimulus, {repair['off_hz']:.1f} Hz without  ({repair['simulations']} simulations)")

entry = {"behaviour": b.name, "attack": attack, "repair": repair, "healthy_on_hz": on, "healthy_off_hz": off,
         "total_seconds": round(time.time() - t0)}
os.makedirs("atlas", exist_ok=True)
json.dump({"entry": entry, "log": log.rows}, open(f"atlas/{b.name}_{args.attacker}.json", "w"), indent=1)
print(f"\n== FRAGILITY ATLAS entry: {b.name} ==")
print(f"  attack size {attack['size']}   repair size {repair['size'] if repair else 'none found'}   "
      f"total time {entry['total_seconds']}s   saved atlas/{b.name}_{args.attacker}.json")
