"""End-to-end test of the Phase 2 / Phase 3 window: drive the real window with scripted keys, then check that every
result it showed on screen is true.

Each scenario runs live.py in a fresh window with --script (keys pressed automatically, each after the previous
result has finished) and --events (every key and every on-screen result logged with the exact brain state). Then:
  1. EXACT: every displayed rate is re-measured independently on the logged brain state -- it must match.
  2. VERDICT: every displayed verdict must follow from that rate and the behaviour's thresholds.
  3. STORY: healthy before the attack, broken after it, back after the repair (bitter: blue team expected to lose).
  4. PHASE 2: no attack or repair may ever happen.
  5. EDGE CASES: repair on a healthy brain, repair twice, reset.

Usage: python virtual_fly/test_window.py [--only eat,drink] [--keep]
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "neuroredteam"))

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="comma-separated scenario names")
ap.add_argument("--analyse_only", action="store_true", help="skip running the windows, re-check existing logs")
args = ap.parse_args()
OUT = os.path.join(HERE, "out", "test_window"); os.makedirs(OUT, exist_ok=True)

G = 1   # seconds of pause after the system is idle again, before the next key
S = lambda *keys: ",".join(f"{k}@{G}" for k in keys) + f",Q@{G}"
SCENARIOS = {   # name: (mode, script, behaviour, the trigger key)
    "eat":     ("phase3", S("S", "A", "S", "D", "S"), "eat", "s"),
    "drink":   ("phase3", S("V", "A", "V", "D", "V"), "drink", "v"),
    "bitter":  ("phase3", S("B", "A", "B", "D", "B"), "refuse_bitter", "b"),
    "escape":  ("phase3", S("L", "A", "L", "D", "L"), "escape", "l"),
    "retreat": ("phase3", S("DOWN", "A", "DOWN", "D", "DOWN"), "retreat", "down"),
    "turn":    ("phase3", S("2", "A", "2", "D", "2"), "turn", "2"),
    "hijack":  ("phase3", S("H", "F", "H", "D", "H"), "eat", "h"),
    "edges":   ("phase3", S("D", "S", "A", "D", "D", "R", "S"), "eat", "s"),
    "phase2":  ("phase2", S("S", "A", "D", "F", "H", "R", "S"), "eat", "s"),
    # the 3-D view controls (O = spin, 0 = front view) must not change any result, in either phase
    "view":    ("phase3", S("S", "O", "A", "D", "0", "S"), "eat", "s"),
    "view2":   ("phase2", S("S", "O", "S", "0", "S"), "eat", "s"),
}
if args.only: SCENARIOS = {k: v for k, v in SCENARIOS.items() if k in args.only.split(",")}

# ------------------------------------------------------------------ run the real window
if not args.analyse_only:
    for name, (mode, script, _, _) in SCENARIOS.items():
        ev = os.path.join(OUT, f"{name}.jsonl")
        if os.path.exists(ev): os.remove(ev)
        cmd = [sys.executable, os.path.join(HERE, "live.py"), "--mode", mode, "--script", script,
               "--events", ev, "--max_seconds", "5400", "--snapshots", os.path.join(OUT, "snapshots")]
        t0 = time.time(); print(f"[{name}] {mode}: {script}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, LIVE_PROFILE="1"))
        open(os.path.join(OUT, f"{name}.stdout.txt"), "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
        print(f"[{name}] exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ check what the screen said
from brain import Brain
from behaviours import Behaviour
brain = Brain("female"); B = {n: Behaviour(brain, n) for n in ("eat", "drink", "refuse_bitter", "escape", "retreat", "turn")}
DNA = {s: int(brain.by_type(["DNa02"], side=s)[0]) for s in ("L", "R")}


def load_iv(iv):
    """event-log brain state -> interventions for Brain.run (JSON turned int keys into strings)"""
    out = {}
    for k, v in (iv or {}).items():
        if k == "lesion": out[k] = [int(x) for x in v]
        elif k == "potentiate": out[k] = {tuple(int(x) for x in key.split(",")): float(f) for key, f in v.items()}
        else: out[k] = {int(a): float(f) for a, f in v.items()}
    return out


HEALTHY = {"eat": "EATS", "drink": "DRINKS", "refuse_bitter": "REFUSES BITTER", "escape": "ESCAPES",
           "retreat": "BACKS UP", "turn": "TURNS LEFT"}
BROKEN = {"eat": {"NO FEEDING"}, "drink": {"NO DRINKING"}, "refuse_bitter": {"EATS BITTER!", "PARTLY CORRUPTED"},
          "escape": {"NO ESCAPE"}, "retreat": {"NO RETREAT"}, "turn": {"NO TURN", "TURNS RIGHT -- INTO THE THREAT"}}


def expected_verdict(name, hz, dL=None, dR=None):
    """what the screen SHOULD say for this rate (mirrors the thresholds, not the window's code)"""
    b = B[name]
    if name == "turn":
        diff = dL - dR
        if abs(diff) < b.on_thr: return "NO TURN"
        return "TURNS LEFT" if diff > 0 else "TURNS RIGHT -- INTO THE THREAT"
    if name == "refuse_bitter":
        return "EATS BITTER!" if hz >= b.on_thr else ("REFUSES BITTER" if hz <= b.off_thr else "PARTLY CORRUPTED")
    return HEALTHY[name] if hz >= b.on_thr else next(iter(BROKEN[name]))


report = {}; all_ok = True
for name, (mode, script, bname, trig) in SCENARIOS.items():
    path = os.path.join(OUT, f"{name}.jsonl")
    fails = []; lines = []
    if not os.path.exists(path):
        report[name] = {"ok": False, "fails": ["no event log -- the window did not run or crashed"]}; all_ok = False; continue
    E = [json.loads(l) for l in open(path, encoding="utf-8")]
    MODIFIERS = ("alt", "ctrl", "shift", "tab", "meta", "windows", "super", "caps")
    keys = [e["key"] for e in E if e["event"] == "key" and not any(m in e["key"] for m in MODIFIERS)]   # Alt-Tab etc.
    expected_keys = [k.lower() for k in script.split(",") for k in [k.split("@")[0]]]
    if keys != expected_keys: fails.append(f"keys pressed {keys} != script {expected_keys}")
    stage = "healthy"; seen_attack = seen_repair = False; replays = []
    # ---- explanation panel and scoreboard (Step 4) ---------------------------------------------------------------
    MODS = ("alt", "ctrl", "shift", "tab", "meta", "windows", "super", "caps")
    for e in E:
        ex = e.get("explain") or ["", "", ""]
        if e["event"] == "after_key" and e["key"] not in ("q",) and not any(m in e["key"] for m in MODS) and not ex[0]:
            fails.append(f"explanation: key '{e['key']}' produced no 'You did' line")
        if e["event"] in ("measured", "threat", "attacked", "repaired", "replay") and e.get("verdict") is not None or e["event"] in ("attacked", "repaired", "replay"):
            is_control = e["event"] == "measured" and not e.get("with_stim") and "NO sugar" not in (e.get("msg") or "")
            if not is_control and not (ex[1] and ex[2]): fails.append(f"explanation: '{e['event']}' left the brain / meaning lines empty: {ex}")
        if e["event"] == "measured" and e.get("with_stim") and e.get("hz") is not None and f"{e['hz']:.0f} spikes" not in ex[1]:
            fails.append(f"explanation: measured {e['hz']:.0f} Hz but the text says: {ex[1]!r}")
        if e["event"] == "threat" and f"left {e['dna_L']:.0f} Hz" not in ex[1]:
            fails.append(f"explanation: threat DNa02 left {e['dna_L']:.0f} Hz but the text says: {ex[1]!r}")
        sc = e.get("score") or []
        if e["event"] == "replay" and e["stage"] == "attack" and not (sc and f"{e['hz']:.0f} Hz" in sc[0]):
            fails.append(f"scoreboard: red line does not show the attacked rate {e['hz']:.0f} Hz: {sc}")
        if e["event"] == "replay" and e["stage"] == "repair" and not (len(sc) > 1 and f"{e['hz']:.0f} Hz" in sc[1]):
            fails.append(f"scoreboard: blue line does not show the repaired rate {e['hz']:.0f} Hz: {sc}")
        if name in ("phase2", "view2") and (sc or any(w in " ".join(ex) for w in ("RED TEAM", "BLUE TEAM"))):
            fails.append(f"PHASE 2 showed red/blue text: {sc or ex}")
    if name == "bitter":
        last_sc = [e.get("score") for e in E if e.get("score")]
        if not last_sc or "cannot repair" not in " ".join(last_sc[-1]): fails.append(f"scoreboard: bitter loss not shown: {last_sc[-1:] }")
    ex_lines = [e.get("explain") for e in E if e["event"] in ("repaired", "replay") and e.get("explain")]
    if ex_lines: lines.append(f"   explanation after the last repair: {ex_lines[-1]}")
    sc_lines = [e.get("score") for e in E if e.get("score")]
    if sc_lines: lines.append(f"   scoreboard at the end: {sc_lines[-1]}")
    for e in E:
        if e["event"] == "after_key": continue                         # screen message right after a key: text checks only
        if e["event"] == "key":
            k = e["key"]
            if k == "a": stage = "after A"
            elif k == "f": stage = "after F"
            elif k == "d" and stage in ("after A", "after F"): stage = "after D"
            elif k == "r": stage = "healthy"
            continue
        if e["event"] == "replay":                                    # the automatic brain replay after A / D
            b = B[e["name"]]; kw = dict(b.sim); kw.update(load_iv(e["iv"])); ws = bool(e["with_stim"])
            r_now = brain.run(b.stim if ws else [], record=b.readouts, seed=1, spikes=True, **kw)
            r_h = brain.run(b.stim if ws else [], record=b.readouts, seed=1, spikes=True, **dict(b.sim))
            hz = float(np.mean([r_now["rates"][i] for i in b.readouts]))
            sil = len(set(np.unique(r_h["spike_i"]).tolist()) - set(np.unique(r_now["spike_i"]).tolist()))
            good = abs(hz - e["hz"]) < 1e-6 and sil == e["n_silenced"]
            lines.append(f"   [replay after {e['stage']:6s}] {e['name']:13s} shown {e['hz']:.1f} Hz, {e['n_silenced']:,} silent  |  "
                         f"independent {hz:.1f} Hz, {sil:,} silent  {'OK' if good else 'MISMATCH'}")
            if not good: fails.append(f"replay after {e['stage']}: shown {e['hz']} Hz / {e['n_silenced']} silent, "
                                      f"independent {hz} Hz / {sil} silent")
            replays.append(e["stage"]); continue
        if e["event"] == "attacked": seen_attack = True; lines.append(f"   attacked: {e['msg']}"); continue
        if e["event"] == "repaired": seen_repair = True; lines.append(f"   repaired: {e['msg']}"); continue
        if e["event"] not in ("measured", "threat") or not e.get("verdict"): continue
        if e["event"] == "measured" and not e["with_stim"] and "HIJACK" not in (e.get("msg") or "") and "NO sugar" not in (e.get("msg") or ""):
            continue                                                   # the no-stimulus control run, not a verdict
        iv = load_iv(e["iv"])
        if e["event"] == "threat":
            stim = brain.by_type(["LC4", "LPLC2"], side=e["side"])
            r = brain.run(stim, record=list(DNA.values()), seed=1, **iv)
            dL, dR = r["rates"][DNA["L"]], r["rates"][DNA["R"]]; shown = (e["dna_L"], e["dna_R"]); true = (dL, dR)
            exp = expected_verdict("turn", None, dL, dR); beh = "turn"
        else:
            beh = e["name"]; b = B[beh]
            kw = dict(b.sim); kw.update(iv)
            true_hz = b.measure(bool(e["with_stim"]), seed=1, **kw)[0]; shown = e["hz"]; true = true_hz
            exp = ("EATS NOTHING" if true_hz >= b.on_thr else "no feeding") if not e["with_stim"] else expected_verdict(beh, true_hz)
        exact = np.allclose(np.atleast_1d(shown), np.atleast_1d(true), atol=1e-6)
        ok_v = e["verdict"] == exp
        lines.append(f"   [{stage:8s}] {beh:13s} shown {np.round(shown, 1)} Hz '{e['verdict']}'  |  independent {np.round(true, 1)} Hz '{exp}'"
                     f"  {'OK' if exact and ok_v else 'MISMATCH'}")
        if not exact: fails.append(f"{stage}: displayed rate {shown} but the same brain state gives {true}")
        if not ok_v: fails.append(f"{stage}: displayed '{e['verdict']}' but the rate means '{exp}'")
        # the story each scenario must tell
        if name in ("phase2", "view2"): continue
        v = e["verdict"]
        if name == "hijack":
            want = {"healthy": "no feeding", "after F": "EATS NOTHING", "after D": "no feeding"}.get(stage)
            if want and v != want: fails.append(f"hijack story: {stage} shows '{v}', expected '{want}'")
        elif stage == "healthy" and v != HEALTHY.get(beh): fails.append(f"story: healthy brain shows '{v}'")
        elif stage == "after A" and v not in BROKEN.get(beh, set()): fails.append(f"story: after the attack still '{v}'")
        elif stage == "after D":
            if beh == "refuse_bitter": pass                         # blue team expected to LOSE (parked issue)
            elif v != HEALTHY.get(beh): fails.append(f"story: after the repair '{v}'")
    msgs = [e.get("msg") or "" for e in E if e["event"] != "key"]
    allmsgs = " | ".join(msgs)
    if name in ("phase2", "view2"):
        if seen_attack or seen_repair or replays: fails.append("PHASE 2 ran an attack, a repair or a replay")
    elif name in ("eat", "drink", "escape", "retreat", "turn", "hijack", "view"):
        if replays != ["attack", "repair"]: fails.append(f"expected a replay after the attack and after the repair, got {replays}")
    elif name == "bitter":
        if replays != ["attack"]: fails.append(f"bitter: expected one replay (after the attack; a lost repair changes nothing), got {replays}")
    if name == "edges":
        for need in ("nothing to repair", "already repaired", "brain reset"):
            if need not in allmsgs:
                fails.append(f"edge case: never showed '{need}'")
        n_rep = sum(e["event"] == "repaired" for e in E)
        if n_rep != 1: fails.append(f"edge case: {n_rep} repairs ran -- exactly one is allowed (D on a healthy brain and a second D must do nothing)")
    if name == "bitter" and seen_repair and "BLUE TEAM LOST" not in allmsgs and "repaired with" in allmsgs:
        lines.append("   note: blue team WON on bitter this time (attack was the repairable kind)")
    if name not in ("phase2", "view2", "edges") and not seen_attack: fails.append("no attack result was ever shown")
    report[name] = {"ok": not fails, "fails": fails, "lines": lines}; all_ok &= not fails

print("\n" + "=" * 110)
for name, r in report.items():
    print(f"{name:8s} {'PASS' if r['ok'] else 'FAIL'}")
    for l in r.get("lines", []): print(l)
    for f in r["fails"]: print(f"   ! {f}")
print("=" * 110)
print("ALL SCENARIOS PASS" if all_ok else "SOME SCENARIOS FAILED -- see above")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
