"""NeuroRedTeam LIVE: a key-driven window. Left: the 3-D fly. Right: its whole connectome, spiking. Bottom: what the
brain decided, the red/blue team log, and the keys.

Keys
  S  drop sugar ahead        V  water ahead      B  bitter (sugar+bitter) ahead      C  choice arena: bitter, water, sugar; visits all three
  E  explore: the fly wanders on its own; mouse click on the fly panel places a drop in that direction
     (left button = sugar, right = bitter, middle = water); whatever it walks into, it tastes
  L  looming shadow (escape)
  DOWN  object approaching head-on (LC16 both eyes -> MDN: backs up)   1 / 2  threat on the left / right eye (-> DNa02: turns away)
  W  walk / stop             <- ->  steer                             SPACE  stop everything, clear drops
  A  red team: attack the current behaviour (block; corrupt if the last drop was bitter)   F  red team: hijack attack (force)
  H  test: does the fly eat with NO sugar?          D  blue team: repair the current behaviour
  R  reset the brain (undo all attacks / repairs)     M  swap female / male brain     Q  quit
Everything the brain does is a real run of the connectome (Brain.run); the body's walking is FlyGym's controller.
Usage: python live.py [--sex female|male] [--camera head|side]
"""
import argparse, os, sys, threading, queue, time, json, hashlib, pickle
import numpy as np, pygame
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "neuroredteam"))
from brain import Brain
from behaviours import Behaviour
from agents import Attacker, Defender, Log, merge_iv, size_of
from body import FlyBody
from brainview import BrainView

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--camera", default="side", choices=["side", "head", "top"])
ap.add_argument("--script", default="", help='auto key presses, e.g. "S@2,A@2,S@2,D@2,S@2,Q@3" (each N seconds after the system is idle again)')
ap.add_argument("--record", default="", help="save the session to this mp4")
ap.add_argument("--robust", action="store_true", help="blue team uses the slow robust defender (default: greedy, fast)")
ap.add_argument("--no_cache", action="store_true", help="always recompute attacks / repairs")
ap.add_argument("--max_seconds", type=float, default=0, help="quit after this many seconds (safety for scripted runs)")
ap.add_argument("--mode", default="full", choices=["full", "phase2"], help="phase2 = behaviours only (no red / blue team keys or panels)")
args = ap.parse_args()

PW, PH, BAR = 640, 480, 200                     # panel size, bottom bar
SIM_PER_FRAME = 0.012                           # simulated seconds per screen frame (~1/8 real time)
ARRIVE = 0.95                                   # mm from drop centre = "at the drop"


class BrainWorker(threading.Thread):
    """Runs connectome jobs off the UI thread. Jobs: ('measure', name, with_stim), ('attack', name), ('repair', name)."""
    def __init__(self, sex):
        super().__init__(daemon=True); self.sex = sex; self.q = queue.Queue(); self.out = queue.Queue(); self.busy = ""
        self.brain = None; self.iv = {}; self.attack = None; self.repair = None; self.start()

    def run(self):
        self.busy = f"loading the {self.sex} brain"
        brain = Brain(self.sex); self.bhv = {n: Behaviour(brain, n) for n in ("eat", "refuse_bitter", "escape", "retreat", "turn", "drink")}
        self.dna02 = {str(brain.types["side"].iloc[i]): int(i) for i in brain.by_type(["DNa02"])}   # {"L": idx, "R": idx}
        self.view = BrainView(brain, PW, PH); self.busy = ""
        self.brain = brain                                     # published last: the UI checks `worker.brain is not None`
        self.out.put(("loaded", None))
        while True:
            job = self.q.get()
            try: self._do(job)
            except Exception as e: self.out.put(("error", repr(e)))
            self.busy = ""

    def _cache_key(self, kind, name, goal):
        h = hashlib.md5(json.dumps(sorted((k, sorted(str(x) for x in (v if isinstance(v, list) else v.items()))) for k, v in self.iv.items())).encode()).hexdigest()[:10]
        return os.path.join(HERE, "cache", f"{self.sex}_{kind}_{name}_{goal}_{'robust' if args.robust else 'greedy'}_{h}.pkl")

    def _do(self, job):
        kind = job[0]
        if kind == "measure":
            _, name, with_stim = job; b = self.bhv[name]; self.busy = f"brain: {name} {'with' if with_stim else 'without'} stimulus"
            kw = dict(b.sim); kw.update(self.iv)
            r = self.brain.run(b.stim if with_stim else [], record=b.readouts, seed=1, spikes=True, **kw)
            hz = float(np.mean([r["rates"][i] for i in b.readouts]))
            self.out.put(("measured", dict(name=name, with_stim=with_stim, hz=hz, spike_t=r["spike_t"], spike_i=r["spike_i"],
                                           on_thr=b.on_thr, off_thr=b.off_thr, expected=b.expected)))
        elif kind == "threat":                                  # looming on one eye: read BOTH steering neurons DNa02
            _, side = job; self.busy = f"brain: threat on the {'left' if side == 'L' else 'right'} eye"
            stim = self.brain.by_type(["LC4", "LPLC2"], side=side); rec = list(self.dna02.values())
            kw = dict(self.iv); r = self.brain.run(stim, record=rec, seed=1, spikes=True, **kw)
            self.out.put(("threat", dict(side=side, dna_L=r["rates"][self.dna02["L"]], dna_R=r["rates"][self.dna02["R"]],
                                         spike_t=r["spike_t"], spike_i=r["spike_i"], on_thr=self.bhv["turn"].on_thr)))
        elif kind == "attack":
            _, name, goal = job; b = self.bhv[name]; self.busy = f"RED TEAM attacking {name} ({goal})"
            ck = self._cache_key("attack", name, goal)
            if not args.no_cache and os.path.exists(ck):
                best, rows = pickle.load(open(ck, "rb")); time.sleep(1.0)              # cached: replay instantly
            else:
                log = Log(quiet=True)
                found = [a for a in (Attacker(b, log, goal, fixed=self.iv).greedy(), Attacker(b, log, goal, fixed=self.iv).inject_inhibitors(),
                                     Attacker(b, log, goal, fixed=self.iv).weaken_only()) if a]
                best = min(found, key=lambda a: (a["size"], a["simulations"])) if found else None
                rows = log.rows; os.makedirs(os.path.dirname(ck), exist_ok=True); pickle.dump((best, rows), open(ck, "wb"))
            if best: self.iv = merge_iv(self.iv, best["intervention"]); self.attack = best
            self.out.put(("attacked", dict(best=best, log=rows, goal=goal)))
        elif kind == "repair":
            _, name = job; b = self.bhv[name]; self.busy = f"BLUE TEAM repairing {name}"
            ck = self._cache_key("repair", name, "block")
            if not args.no_cache and os.path.exists(ck):
                rep, rows = pickle.load(open(ck, "rb")); time.sleep(1.0)
            else:
                log = Log(quiet=True); D = Defender(b, {"intervention": self.iv}, log)
                rep = D.robust() if args.robust else D.greedy()
                rows = log.rows; os.makedirs(os.path.dirname(ck), exist_ok=True); pickle.dump((rep, rows), open(ck, "wb"))
            if rep: self.iv = merge_iv(self.iv, rep["intervention"]); self.repair = rep
            self.out.put(("repaired", dict(rep=rep, log=rows)))
        elif kind == "reset":
            self.iv = {}; self.attack = None; self.repair = None; self.out.put(("reset", None))


def neurons_of(iv):
    if not iv: return []
    return sorted(set(iv.get("lesion", [])) | set(iv.get("inject", {})) | set(iv.get("weaken", {})) | set(iv.get("bias", {}))
                  | {a for (a, b) in iv.get("potentiate", {})})


pygame.init(); screen = pygame.display.set_mode((2 * PW, PH + BAR)); pygame.display.set_caption("The fly and its brain" if args.mode == "phase2" else "NeuroRedTeam LIVE")
font = pygame.font.SysFont("consolas", 16); big = pygame.font.SysFont("arial", 34, bold=True)
clock = pygame.time.Clock()
worker = BrainWorker(args.sex)
body = FlyBody(camera=args.camera, sugar_at=(1e6, 1e6), bitter_at=(1e6, 1e6), water_at=(1e6, 1e6), buffer=False)   # drops parked off-screen
geoms = {"sugar": body.sim.mj_model.geom("sugar"), "bitter": body.sim.mj_model.geom("bitter"), "water": body.sim.mj_model.geom("water")}

LABELS = {"eat": ("sugar taste neurons", "MN9 (feeding muscle)"), "drink": ("water taste neurons", "MN9 (feeding muscle)"),
          "refuse_bitter": ("sugar + bitter taste neurons", "MN9 (feeding muscle)"), "escape": ("looming detectors LC4/LPLC2", "giant fiber DNp01 (escape)"),
          "retreat": ("approach detectors LC16, both eyes", "MDN moonwalker (walk backward)"), "turn": ("looming detectors, one eye", "DNa02 (steering)")}

state = dict(walking=False, steer=0.0, target=None, kind=None, tour=[], spots={}, explore=False, wander=0.0, drops={}, phase="idle", t_phase=0.0, eat=0.0, turn_dir=0.0, verdict=None, verdict_col=(255, 255, 255),
             msg="loading the brain ...", loglines=[], last_frame=None, brain_img=None, anim=None)


def place_drop(kind):
    pos = body.position[:2]; h = float(body.heading)
    tgt = pos + 9.0 * np.array([np.cos(h), np.sin(h)])
    for k_, g in geoms.items(): body.sim.mj_model.geom_pos[g.id][:2] = tgt if k_ == kind else (1e6, 1e6)
    state["drops"] = {kind: tgt}
    state.update(target=tgt, kind=kind, walking=True, phase="approach", verdict=None)
    state["msg"] = f"{kind} placed 9 mm ahead; walking to it"


def place_arena():
    """Choice arena: bitter (left), water (centre), sugar (right) laid out ahead; the fly visits them in that order."""
    pos = body.position[:2]; h = float(body.heading)
    order = [("bitter", 7.0, +0.55), ("water", 8.5, 0.0), ("sugar", 7.0, -0.55)]      # (kind, distance mm, angle rad)
    spots = {}
    for kind, dist, ang in order:
        spots[kind] = pos + dist * np.array([np.cos(h + ang), np.sin(h + ang)])
        body.sim.mj_model.geom_pos[geoms[kind].id][:2] = spots[kind]
    state["drops"] = dict(spots)
    state.update(tour=[k for k, _, _ in order][1:], spots=spots, target=spots["bitter"], kind="bitter", walking=True, phase="approach", verdict=None)
    state["msg"] = "CHOICE: bitter, water and sugar drops ahead; the fly will taste each one"


def next_stop():
    """After a taste in the arena, walk to the next drop (or finish)."""
    st = state
    if st.get("tour"):
        k = st["tour"].pop(0); st.update(target=st["spots"][k], kind=k, walking=True, phase="approach", verdict=None)
        st["msg"] = f"next: the {k} drop"; return True
    return False


def drop_at(kind, angle, dist=8.0):
    """Place a drop `dist` mm from the fly at `angle` rad relative to its heading (mouse: direction of the click)."""
    pos = body.position[:2]; h = float(body.heading)
    spot = pos + dist * np.array([np.cos(h + angle), np.sin(h + angle)])
    body.sim.mj_model.geom_pos[geoms[kind].id][:2] = spot; state["drops"][kind] = spot
    state["msg"] = f"{kind} drop placed {'ahead' if abs(angle) < 0.3 else ('to the left' if angle > 0 else 'to the right')}; exploring fly will find it"


def remove_drop(kind):
    body.sim.mj_model.geom_pos[geoms[kind].id][:2] = (1e6, 1e6); state["drops"].pop(kind, None)


def start_verdict(name):
    state.update(phase="think", t_phase=0.0); worker.q.put(("measure", name, True)); worker.q.put(("measure", name, False))


def draw_text(lines, x, y, col=(230, 230, 230)):
    for i, l in enumerate(lines): screen.blit(font.render(l, True, col), (x, y + 18 * i))


running = True; escape_kick = 0.0; t_loaded = None; frames_out = []; t_start = time.time()
script = [(float(x.split("@")[1]), x.split("@")[0].strip().upper()) for x in args.script.split(",") if "@" in x]
while running:
    # scripted keys: each fires N seconds after the system went idle (brain done, fly not busy) following the previous key
    idle = worker.brain is not None and worker.q.empty() and worker.busy == "" and state["phase"] in ("idle", "free", "explore") and state["anim"] is None
    if args.max_seconds and time.time() - t_start > args.max_seconds: running = False       # safety: a script can never hang
    if idle and t_loaded is None: t_loaded = time.time()
    if not idle: t_loaded = None
    if t_loaded is not None and script and time.time() - t_loaded >= script[0][0]:
        _, key = script.pop(0); t_loaded = None
        if key in ("LCLICK", "RCLICK", "MCLICK"):                    # scripted mouse clicks on the fly panel (for tests / clips)
            btn = {"LCLICK": 1, "RCLICK": 3, "MCLICK": 2}[key]; x = {"LCLICK": int(PW * 0.25), "RCLICK": int(PW * 0.75), "MCLICK": PW // 2}[key]
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=btn, pos=(x, PH // 2)))
        else:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{key.lower()}") if len(key) == 1 else getattr(pygame, f"K_{key}")))
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running = False
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_q: running = False
            elif worker.brain is None: pass
            elif k == pygame.K_s: place_drop("sugar")
            elif k == pygame.K_b: place_drop("bitter")
            elif k == pygame.K_v: place_drop("water")
            elif k == pygame.K_c: place_arena()
            elif k == pygame.K_e:
                state["explore"] = not state["explore"]
                if state["explore"]: state.update(phase="explore", walking=True, target=None, tour=[], verdict=None); state["msg"] = "EXPLORE: the fly wanders; click to place drops (left = sugar, right = bitter, middle = water)"
                else: state.update(phase="idle", walking=False); state["msg"] = "explore off"
        elif ev.type == pygame.MOUSEBUTTONDOWN and worker.brain is not None and ev.pos[0] < PW:
            kind = {1: "sugar", 3: "bitter", 2: "water"}.get(ev.button)
            if kind:
                angle = (PW / 2 - ev.pos[0]) / (PW / 2) * 1.0            # click left of centre -> to the fly's left (up to ~57 deg)
                drop_at(kind, angle)
                if not state["explore"]: state.update(target=state["drops"][kind], kind=kind, walking=True, phase="approach", verdict=None)
            elif k == pygame.K_w: state["walking"] = not state["walking"]; state["phase"] = "free"; state["msg"] = "walking" if state["walking"] else "stopped"
            elif k == pygame.K_LEFT: state["steer"] = min(state["steer"] + 0.4, 0.8)
            elif k == pygame.K_RIGHT: state["steer"] = max(state["steer"] - 0.4, -0.8)
            elif k == pygame.K_SPACE: state.update(walking=False, steer=0.0, target=None, phase="idle", verdict=None, tour=[], explore=False); state["msg"] = "stopped"
            elif k == pygame.K_l: state.update(phase="think", kind="looming", walking=False); state["msg"] = "looming shadow! brain: escape?"; worker.q.put(("measure", "escape", True))
            elif k == pygame.K_DOWN: state.update(phase="think", kind="approach", walking=False); state["msg"] = "object approaching head-on (LC16, both eyes): brain: back up?"; worker.q.put(("measure", "retreat", True))
            elif k in (pygame.K_1, pygame.K_2):
                side = "L" if k == pygame.K_1 else "R"; state.update(phase="think", kind="threat", walking=False)
                state["msg"] = f"threat on the {'LEFT' if side == 'L' else 'RIGHT'} eye: brain: which steering neuron fires?"; worker.q.put(("threat", side))
            elif k == pygame.K_h: state.update(phase="think", kind="hijack", walking=False); state["msg"] = "HIJACK: no sugar at all; does the attacked/forced brain eat?"; worker.q.put(("measure", "eat", False))
            elif args.mode == "phase2" and k in (pygame.K_a, pygame.K_f, pygame.K_h, pygame.K_d, pygame.K_r): pass
            elif k == pygame.K_a:
                name = "refuse_bitter" if state["kind"] == "bitter" else ("escape" if state["kind"] == "looming" else "eat")
                goal = "corrupt" if name == "refuse_bitter" else "block"
                state["msg"] = f"RED TEAM: attacking {name} ..."; worker.q.put(("attack", name, goal))
            elif k == pygame.K_f:
                state["msg"] = "RED TEAM: hijack -- make the fly eat with NO sugar (goal force) ..."; state["kind"] = "sugar"; worker.q.put(("attack", "eat", "force"))
            elif k == pygame.K_d:
                name = "escape" if state["kind"] == "looming" else "eat"
                state["msg"] = f"BLUE TEAM: repairing {name} ..."; worker.q.put(("repair", name))
            elif k == pygame.K_r: worker.q.put(("reset",)); state["msg"] = "brain reset: no attacks, no repairs"
            elif k == pygame.K_m:
                worker = BrainWorker("male" if worker.sex == "female" else "female"); state["msg"] = f"loading the {worker.sex} brain ..."

    # ---- brain results
    try:
        while True:
            kind, data = worker.out.get_nowait()
            if kind == "loaded": state["msg"] = f"{worker.sex} brain ready: {worker.brain.N:,} neurons. S = sugar, B = bitter, L = looming, A/D = attack/repair"
            elif kind == "measured":
                d = data; st = state
                if d["with_stim"] or st["kind"] == "hijack":        # draw the stimulus run; the no-stimulus control is only logged
                    st["anim"] = dict(t0=time.time(), spike_t=d["spike_t"], spike_i=d["spike_i"], dur=3.0, fresh=True)
                st["loglines"].append(f"brain run: {d['name']} {'WITH' if d['with_stim'] else 'without'} stimulus -> readout {d['hz']:.0f} Hz")
                if d["name"] == "escape" and d["with_stim"]:
                    ok = d["hz"] >= d["on_thr"]; st["verdict"] = "ESCAPES" if ok else "NO ESCAPE"; st["verdict_col"] = (120, 230, 120) if ok else (240, 110, 110)
                    st["phase"] = "escape" if ok else "idle"; escape_kick = 1.0 if ok else 0.0; st["t_phase"] = 0.0
                    st["msg"] = f"giant fiber DNp01 at {d['hz']:.0f} Hz (bar {d['on_thr']:.0f}) -> {st['verdict']}"
                elif d["name"] == "retreat" and d["with_stim"]:
                    ok = d["hz"] >= d["on_thr"]; st["verdict"] = "BACKS UP" if ok else "NO RETREAT"; st["verdict_col"] = (120, 230, 120) if ok else (240, 110, 110)
                    st["phase"] = "retreat" if ok else "idle"; st["t_phase"] = 0.0
                    st["msg"] = f"moonwalker neurons MDN at {d['hz']:.0f} Hz (bar {d['on_thr']:.0f}) -> {st['verdict']}"
                elif d["name"] == "eat" and not d["with_stim"] and st["kind"] == "hijack":
                    ok = d["hz"] >= d["on_thr"]; st["verdict"] = "EATS NOTHING" if ok else "no feeding"; st["verdict_col"] = (240, 200, 90) if ok else (200, 200, 200)
                    st["phase"] = "eating" if ok else "idle"; st["t_phase"] = 0.0
                    st["msg"] = f"MN9 with NO sugar: {d['hz']:.0f} Hz -> {st['verdict']}"
                elif d["with_stim"] and st["kind"] in ("sugar", "bitter", "water"):
                    ok = d["hz"] >= d["on_thr"]
                    if st["kind"] == "sugar": st["verdict"] = "EATS" if ok else "NO FEEDING"; st["verdict_col"] = (120, 230, 120) if ok else (240, 110, 110)
                    elif st["kind"] == "water": st["verdict"] = "DRINKS" if ok else "NO DRINKING"; st["verdict_col"] = (120, 200, 255) if ok else (240, 110, 110)
                    else: st["verdict"] = "EATS BITTER!" if ok else "REFUSES BITTER"; st["verdict_col"] = (240, 110, 110) if ok else (120, 230, 120)
                    st["phase"] = "eating" if ok else "refuse"; st["t_phase"] = 0.0
                    st["msg"] = f"MN9 {d['hz']:.0f} Hz with {st['kind']} (bar {d['on_thr']:.0f}) -> {st['verdict']}"
                elif not d["with_stim"]:
                    st["loglines"].append(f"   (control without stimulus: {d['hz']:.0f} Hz, must be <= {d['off_thr']:.0f})")
            elif kind == "threat":
                d = data; st = state; st["anim"] = dict(t0=time.time(), spike_t=d["spike_t"], spike_i=d["spike_i"], dur=3.0, fresh=True)
                diff = d["dna_L"] - d["dna_R"]                          # steering neurons fire asymmetrically -> a turn
                if abs(diff) >= d["on_thr"]:                             # direction: away from the threatened eye
                    away = -1.0 if d["side"] == "L" else 1.0             # +1 = turn left
                    st["verdict"] = "TURNS RIGHT" if away < 0 else "TURNS LEFT"; st["verdict_col"] = (120, 230, 120)
                    st["phase"] = "turn"; st["t_phase"] = 0.0; st["turn_dir"] = away
                else: st["verdict"] = "NO TURN"; st["verdict_col"] = (240, 110, 110); st["phase"] = "idle"
                st["loglines"].append(f"brain run: threat on {d['side']} eye -> DNa02 left {d['dna_L']:.0f} Hz, right {d['dna_R']:.0f} Hz")
                st["msg"] = f"steering neurons: DNa02 L {d['dna_L']:.0f} Hz, R {d['dna_R']:.0f} Hz (bar {d['on_thr']:.0f}) -> {st['verdict']}" + (" (away from the threat)" if st["phase"] == "turn" else "")
            elif kind == "attacked":
                b_ = data["best"]
                for row in data["log"][-6:]: state["loglines"].append(f"  [red {row['step']:2d}] {'GOAL ' if row['goal_met'] else '     '}{row['hypothesis'][:48]:48s} -> {row['result']}")
                state["msg"] = (f"RED TEAM: {data['goal']} with {b_['size']} neurons ({b_['strategy'].replace('attacker-', '')}). Press S/B/L to test the fly."
                                if b_ else "RED TEAM: no attack found within budget")
            elif kind == "repaired":
                r_ = data["rep"]
                for row in data["log"][-6:]: state["loglines"].append(f"  [blue {row['step']:2d}] {'GOAL ' if row['goal_met'] else '     '}{row['hypothesis'][:48]:48s} -> {row['result']}")
                state["msg"] = (f"BLUE TEAM: repaired with {r_['size']} neurons ({r_['move']}), {r_['on_hz']:.0f} Hz{' [fragile]' if r_.get('fragile') else ''}. Press S to test."
                                if r_ else "BLUE TEAM: no repair found within budget")
            elif kind == "reset": state["msg"] = "brain reset"
            elif kind == "error": state["msg"] = "error: " + data
            state["loglines"] = state["loglines"][-7:]
    except queue.Empty: pass

    # ---- body
    st = state; left = right = 0.0; eat = 0.0
    if st["phase"] == "approach" and st["target"] is not None:
        d = st["target"] - body.position[:2]; dist = np.linalg.norm(d)
        if dist < ARRIVE:
            st["walking"] = False; start_verdict({"sugar": "eat", "bitter": "refuse_bitter", "water": "drink"}[st["kind"]])
            st["msg"] = f"at the {st['kind']} drop; asking the brain ..."
        else:
            err = (np.arctan2(d[1], d[0]) - body.heading + np.pi) % (2 * np.pi) - np.pi
            left, right = np.clip(1.0 - 0.8 * err, 0.2, 1.2), np.clip(1.0 + 0.8 * err, 0.2, 1.2)
    elif st["phase"] == "explore":
        # slow random walk: the steering bias drifts, and the fly turns back toward the arena centre when far out
        st["wander"] = 0.97 * st["wander"] + 0.06 * np.random.randn()
        pos = body.position[:2]; far = np.linalg.norm(pos)
        if far > 18.0:
            back = (np.arctan2(-pos[1], -pos[0]) - body.heading + np.pi) % (2 * np.pi) - np.pi
            st["wander"] = 0.6 * np.sign(back)
        # "smell": within 12 mm of a drop the wander steers toward the nearest one (a body-side rule, not the connectome)
        near = [(np.linalg.norm(spot - pos), kind, spot) for kind, spot in st["drops"].items() if np.linalg.norm(spot - pos) < 12.0]
        if near:
            dist, kind, spot = min(near)
            err = (np.arctan2(*(spot - pos)[::-1]) - body.heading + np.pi) % (2 * np.pi) - np.pi
            st["wander"] = 0.7 * st["wander"] + 0.3 * np.clip(0.9 * err, -0.6, 0.6)
            if st["msg"].startswith("EXPLORE") or "placed" in st["msg"]: st["msg"] = f"smells the {kind} drop ({dist:.0f} mm; steering rule, not the connectome)"
        w = float(np.clip(st["wander"], -0.6, 0.6)); left, right = np.clip(0.9 - w, 0.2, 1.2), np.clip(0.9 + w, 0.2, 1.2)
        for kind, spot in list(st["drops"].items()):                  # anything within reach gets tasted
            if np.linalg.norm(spot - pos) < ARRIVE:
                st.update(kind=kind, walking=False); start_verdict({"sugar": "eat", "bitter": "refuse_bitter", "water": "drink"}[kind])
                st["msg"] = f"found the {kind} drop; asking the brain ..."; break
    elif st["phase"] == "free" and st["walking"]:
        left, right = np.clip(1.0 - st["steer"], 0.2, 1.2), np.clip(1.0 + st["steer"], 0.2, 1.2)
    elif st["phase"] == "eating":
        eat = 1.0; st["t_phase"] += SIM_PER_FRAME
        if st["t_phase"] > 2.5:
            st["phase"] = "idle"
            if st["explore"]: remove_drop(st["kind"]); st["phase"] = "explore"; st["walking"] = True; st["verdict"] = None
            else: next_stop()
    elif st["phase"] == "refuse":
        st["t_phase"] += SIM_PER_FRAME
        if st["t_phase"] > 1.0 and st["explore"]: remove_drop(st["kind"]); st["phase"] = "explore"; st["walking"] = True; st["verdict"] = None
        elif st["t_phase"] > 1.0 and st.get("tour"): next_stop()                                   # in the arena: go to the next drop
        elif st["t_phase"] > 1.0: st["phase"] = "free"; st["walking"] = True; st["steer"] = 0.6    # alone: walks away, turning
        if st["t_phase"] > 3.0: st["steer"] = 0.0
    elif st["phase"] == "escape":
        st["t_phase"] += SIM_PER_FRAME
        if st["t_phase"] > 1.2: st["phase"] = "idle"
    elif st["phase"] == "retreat":                              # MDN: walk backwards (negative descending drive)
        left = right = -0.8; st["t_phase"] += SIM_PER_FRAME
        if st["t_phase"] > 1.5: st["phase"] = "idle"
    elif st["phase"] == "turn":                                 # DNa02 asymmetry: turn toward the more active side
        d_ = st["turn_dir"]; left, right = 1.0 - 0.7 * d_, 1.0 + 0.7 * d_; st["t_phase"] += SIM_PER_FRAME
        if st["t_phase"] > 1.5: st["phase"] = "idle"
    n = int(SIM_PER_FRAME / body.timestep)
    for i in range(n):
        if escape_kick > 0: body.kick(escape_kick); escape_kick = 0.0
        body.step(left, right, eat=eat, render=False)
    fly_img = body.frame()

    # ---- brain panel
    bimg = None
    if worker.brain is not None:
        v = worker.view; anim = st["anim"]
        if anim is not None:
            if anim.get("fresh"): v.reset(); anim["fresh"] = False
            frac = min((time.time() - anim["t0"]) / anim["dur"], 1.0); v.advance(300.0 * frac, anim["spike_t"], anim["spike_i"])
            if frac >= 1.0: st["anim"] = None
        else: v.glow *= 0.95
        cur_name = {"looming": "escape", "bitter": "refuse_bitter", "approach": "retreat", "threat": "turn", "water": "drink"}.get(st["kind"], "eat")
        cur = worker.bhv[cur_name]
        stim_label, readout_label = LABELS[cur_name]
        att = neurons_of(worker.attack["intervention"]) if worker.attack else []
        rep = neurons_of(worker.repair["intervention"]) if worker.repair else []
        bimg = v.frame(stimulus=cur.stim, readout=cur.readout, attacked=att, repaired=rep, stim_label=stim_label, readout_label=readout_label,
                       show_redblue=(args.mode != "phase2"), readout_bar=cur.on_thr,
                       title=f"{worker.sex} fly connectome: {worker.brain.N:,} neurons", subtitle=(worker.busy or "idle"),
                       stim_on=st["phase"] in ("think", "eating", "refuse", "escape", "retreat", "turn"), blink=0.5 + 0.5 * np.sin(time.time() * 4))

    # ---- draw
    screen.fill((18, 18, 24))
    surf = pygame.surfarray.make_surface(np.transpose(fly_img, (1, 0, 2))); surf = pygame.transform.smoothscale(surf, (PW, PH)); screen.blit(surf, (0, 0))
    if bimg is not None: screen.blit(pygame.surfarray.make_surface(np.transpose(bimg, (1, 0, 2))), (PW, 0))
    else: draw_text(["loading the brain (female ~10 s, male ~40 s) ..."], PW + 20, PH // 2)
    iv_txt = f"brain state: {'healthy' if not worker.iv else f'{size_of(worker.iv)} neurons modified'}" + (f"  attack {worker.attack['size']}" if worker.attack else "") + (f"  repair {worker.repair['size']}" if worker.repair else "")
    if args.mode != "phase2": draw_text([iv_txt], 10, 10, (20, 20, 20))
    if st["verdict"]:
        t = big.render(st["verdict"], True, (0, 0, 0)); pygame.draw.rect(screen, st["verdict_col"], (PW - t.get_width() - 30, 8, t.get_width() + 20, t.get_height() + 6)); screen.blit(t, (PW - t.get_width() - 20, 10))
    draw_text([st["msg"]], 10, PH + 8, (255, 230, 120))
    draw_text(st["loglines"], 10, PH + 30, (200, 200, 210))
    keys_p2 = "E explore + click (L sugar, R bitter, M water)   S V B drops   C arena   L looming   DOWN approach   1/2 threat   W walk   SPACE stop   Q"
    draw_text([keys_p2 + ("" if args.mode == "phase2" else "  |  A attack  F hijack  H no-sugar test  D repair  R reset")], 10, PH + BAR - 22, (140, 140, 160))
    pygame.display.flip(); clock.tick(30)
    if args.record: frames_out.append(np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)).copy())
pygame.quit()
if args.record and frames_out:
    import imageio.v2 as iio
    iio.mimwrite(args.record, frames_out[::1], fps=30, codec="libx264", quality=8); print("saved", args.record, len(frames_out), "frames")
