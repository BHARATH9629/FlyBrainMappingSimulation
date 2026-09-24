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
CODE_VERSION = hashlib.md5(b"".join(open(os.path.join(HERE, "..", "neuroredteam", f), "rb").read()
                                    for f in ("agents.py", "behaviours.py", "brain.py"))).hexdigest()[:8]
from brain import Brain
from behaviours import Behaviour, BEHAVIOURS
from agents import Attacker, Defender, Log, merge_iv, size_of
from body import FlyBody
from brainview import BrainView
from explain import explain_key, explain_result, score_lines, SHORT

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--camera", default="side", choices=["side", "head", "top"])
ap.add_argument("--script", default="", help='auto key presses, e.g. "S@2,A@2,S@2,D@2,S@2,Q@3" (each N seconds after the system is idle again)')
ap.add_argument("--record", default="", help="save the session to this mp4")
ap.add_argument("--robust", action="store_true", help="blue team uses the slow robust defender (default: greedy, fast)")
ap.add_argument("--no_cache", action="store_true", help="always recompute attacks / repairs")
ap.add_argument("--max_seconds", type=float, default=0, help="quit after this many seconds (safety for scripted runs)")
ap.add_argument("--snapshots", default="", help="save the brain panel at the end of every attack / repair replay to this folder (tests)")
ap.add_argument("--events", default="", help="append every key press and every on-screen result to this JSONL file (tests)")
ap.add_argument("--windowed", action="store_true", help="a window instead of the full screen (scripted runs are always windowed)")
ap.add_argument("--mode", default="full", choices=["full", "phase2", "phase3"],
                help="phase2 = behaviours only (every red/blue key refused); phase3 = the attack/repair demo; full = both")
args = ap.parse_args()


def log_event(kind, **fields):
    """One JSON line per key press / on-screen result, for the end-to-end window test (tests only, off by default)."""
    if not args.events: return
    def plain(v):
        if isinstance(v, dict): return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [plain(x) for x in v]
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    with open(args.events, "a", encoding="utf-8") as f:
        f.write(json.dumps(plain(dict(event=kind, t=round(time.time(), 2), **fields))) + "\n")

W, H = 1280, 800                                # the canvas: everything is drawn at this size, then fills the screen
PW, PH = 622, 418                               # the fly picture and the brain picture (inside their cards)
FLY_H = 466                                     # the fly camera renders 640 x 480 -> 622 x 466; the card shows its middle 418 rows
HDR = 30                                        # card title strip
FLY_CARD, BRAIN_CARD = pygame.Rect(12, 98, PW, HDR + PH), pygame.Rect(646, 98, PW, HDR + PH)
FLY_C, BRAIN_C = pygame.Rect(12, 98 + HDR, PW, PH), pygame.Rect(646, 98 + HDR, PW, PH)     # the pictures themselves
STORY_CARD, SCORE_CARD = pygame.Rect(12, 556, 770, 186), pygame.Rect(794, 556, 474, 186)
STRIP_Y, KEYS_Y = 58, 752
BG, PANEL, INK, DIM, ACCENT = (13, 14, 19), (24, 26, 34), (236, 237, 242), (146, 150, 166), (255, 214, 102)
EDGE, RED, BLUE, GREEN = (46, 50, 66), (255, 96, 86), (110, 160, 255), (110, 215, 140)
SIM_PER_FRAME = 0.012                           # simulated seconds per screen frame (~1/8 real time)
ARRIVE = 0.95                                   # mm from drop centre = "at the drop"


class BrainWorker(threading.Thread):
    """Runs connectome jobs off the UI thread. Jobs: ('measure', name, with_stim), ('attack', name), ('repair', name)."""
    def __init__(self, sex):
        super().__init__(daemon=True); self.sex = sex; self.q = queue.Queue(); self.out = queue.Queue(); self.busy = ""
        self.brain = None; self.iv = {}; self.attack = None; self.repair = None; self.attack_name = None; self.attack_goal = None
        self.healthy_fired = {}                     # (behaviour, with_stim) -> neurons that fire on the UNMODIFIED brain
        self.healthy_hz = {}                        # (behaviour, with_stim) -> readout rate of the UNMODIFIED brain
        self.healthy_spk = {}                       # (behaviour, with_stim) -> spikes of the UNMODIFIED brain (the 'healthy' picture)
        self.start()

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
        # the agents' own code is part of the key: a cached result is never replayed after the agents' code changes
        h += "_" + CODE_VERSION
        return os.path.join(HERE, "cache", f"{self.sex}_{kind}_{name}_{goal}_{'robust' if args.robust else 'greedy'}_{h}.pkl")

    def _replay(self, name, stage, with_stim):
        """Run the behaviour once on the brain as it now stands (after an attack or a repair) and send the spikes to the
        brain panel, with which neurons went silent compared with the healthy brain. Same call as the 'measure' job."""
        b = self.bhv[name]; stim = b.stim if with_stim else []
        kw = dict(b.sim); kw.update(self.iv)
        r = self.brain.run(stim, record=b.readouts, seed=1, spikes=True, **kw)
        key = (name, bool(with_stim))
        if key not in self.healthy_fired:
            r0 = self.brain.run(stim, record=b.readouts, seed=1, spikes=True, **dict(b.sim))
            self.healthy_fired[key] = set(np.unique(r0["spike_i"]).tolist())
            self.healthy_hz[key] = float(np.mean([r0["rates"][i] for i in b.readouts]))
            self.healthy_spk[key] = (r0["spike_t"], r0["spike_i"])
        now = set(np.unique(r["spike_i"]).tolist()); healthy = self.healthy_fired[key]
        self.out.put(("replay", dict(name=name, stage=stage, with_stim=with_stim,
                                     hz=float(np.mean([r["rates"][i] for i in b.readouts])),
                                     spike_t=r["spike_t"], spike_i=r["spike_i"],
                                     silenced=np.array(sorted(healthy - now), int), n_fired=len(now),
                                     n_silenced=len(healthy - now), n_new=len(now - healthy),
                                     on_thr=b.on_thr, off_thr=b.off_thr, healthy=self.healthy_hz[key],
                                     h_spike_t=self.healthy_spk[key][0], h_spike_i=self.healthy_spk[key][1])))

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
            if best:
                self.iv = merge_iv(self.iv, best["intervention"]); self.attack = best; self.attack_name = name; self.attack_goal = goal
                self.repair = None                          # a new attack re-opens the brain for repair
            self.out.put(("attacked", dict(best=best, log=rows, goal=goal)))
            if best: self._replay(name, "attack", with_stim=(goal != "force"))
        elif kind == "repair":
            _, name, goal = (job + ("block",))[:3]; b = self.bhv[name]; self.busy = f"BLUE TEAM repairing {name}"
            ck = self._cache_key("repair", name, goal)          # the goal MUST be in the key: a corrupt repair and a
            if not args.no_cache and os.path.exists(ck):
                rep, rows = pickle.load(open(ck, "rb")); time.sleep(1.0)
            else:
                log = Log(quiet=True); D = Defender(b, {"intervention": self.iv}, log)
                if goal == "corrupt":
                    # restoring a REFUSAL means pushing the readout DOWN, which restore_off() does (greedy() pushes it UP)
                    ctrl = BEHAVIOURS[self.sex][name].get("control")
                    rep = D.restore_off(control=self.bhv[ctrl] if ctrl in self.bhv else None)
                elif args.robust: rep = D.robust()
                else:
                    # restore(): bring the behaviour back to >= 80 % of healthy on every seed
                    rep = D.restore(target=0.8)       # below 80 %: restore() returns its strongest working repair itself
                rows = log.rows; os.makedirs(os.path.dirname(ck), exist_ok=True); pickle.dump((rep, rows), open(ck, "wb"))
            # a PARTIAL result is the blue team losing: never apply it to the brain and never call it a repair
            applied = bool(rep) and not rep.get("partial")
            if applied: self.iv = merge_iv(self.iv, rep["intervention"]); self.repair = rep
            self.out.put(("repaired", dict(rep=rep, log=rows)))
            if applied: self._replay(name, "repair", with_stim=(self.attack_goal != "force"))
        elif kind == "reset":
            self.iv = {}; self.attack = None; self.repair = None; self.attack_name = None; self.attack_goal = None; self.out.put(("reset", None))


def run_counts(spike_i):
    """neurons that fired in a run, and how many of them have no recorded position (reported, never hidden)"""
    u = np.unique(spike_i)
    return dict(fired=int(u.size), missing=int((~worker.view.ok[u]).sum()) if u.size else 0)


def neurons_of(iv):
    if not iv: return []
    return sorted(set(iv.get("lesion", [])) | set(iv.get("inject", {})) | set(iv.get("weaken", {})) | set(iv.get("bias", {}))
                  | {a for (a, b) in iv.get("potentiate", {})})


pygame.init()
pygame.display.set_caption("The fly and its brain" if args.mode == "phase2" else ("NeuroRedTeam -- PHASE 3: red team vs blue team" if args.mode == "phase3" else "NeuroRedTeam LIVE"))


def open_display(full):
    """full screen at the screen's own size, or a window that fits on it; the canvas is scaled into either"""
    ds = pygame.display.get_desktop_sizes()[0]
    if full: return pygame.display.set_mode(ds, pygame.FULLSCREEN)
    k = min(1.0, (ds[0] - 16) / W, (ds[1] - 90) / H)
    return pygame.display.set_mode((int(W * k), int(H * k)))


FULL = not (args.script or args.windowed)
display = open_display(FULL); screen = pygame.Surface((W, H))


def _fit():
    dw, dh = display.get_size(); k = min(dw / W, dh / H)
    return k, (dw - int(W * k)) // 2, (dh - int(H * k)) // 2


def present():
    k, ox, oy = _fit()
    if k == 1.0: display.blit(screen, (ox, oy))
    else: display.fill(BG); display.blit(pygame.transform.smoothscale(screen, (int(W * k), int(H * k))), (ox, oy))
    pygame.display.flip()


def to_canvas(ev_or_pos):
    """screen pixel -> canvas pixel (scripted clicks are already in canvas pixels)"""
    if getattr(ev_or_pos, "scripted", False): return ev_or_pos.pos
    pos = getattr(ev_or_pos, "pos", ev_or_pos); k, ox, oy = _fit()
    return (int((pos[0] - ox) / k), int((pos[1] - oy) / k))


F = lambda size, bold=False: pygame.font.SysFont("segoeui", size, bold=bold)
font, small, cap, head, title_font, big = F(17), F(14), F(12, True), F(19, True), F(24, True), F(34, True)
body_f = F(16)                                  # the explanation text
HEAD_FONTS = (head, F(17, True), F(15, True))   # the headline: largest that fits
mono = pygame.font.SysFont("consolas", 14)


def rrect(surface, color, rect, r=8, width=0):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=r)


def _corner_masks(r=12, col=BG):
    """small patches that round off the bottom corners of a picture (outside the quarter circle: background colour)"""
    out = []
    for flip in (False, True):
        m = pygame.Surface((r, r), pygame.SRCALPHA)
        for y in range(r):
            for x in range(r):
                cx, cy = (x + 0.5, y + 0.5) if flip else (r - x - 0.5, y + 0.5)     # distance from the arc's centre
                if cx * cx + cy * cy > r * r: m.set_at((x, y), col)
        out.append(m)
    return out
CORNER_L, CORNER_R = _corner_masks()


def card(rect, title, right="", accent=None):
    """a rounded panel with a small caps title strip; returns the inner content top y"""
    rrect(screen, (6, 7, 10), rect.move(0, 3), r=12)                             # soft shadow
    rrect(screen, PANEL, rect, r=12)
    if accent: pygame.draw.rect(screen, accent, (rect.x + 14, rect.y + 11, 3, 12), border_radius=2)
    screen.blit(cap.render(title.upper(), True, (178, 182, 198)), (rect.x + (22 if accent else 14), rect.y + 9))
    if right:
        t = small.render(right, True, DIM); screen.blit(t, (rect.right - t.get_width() - 14, rect.y + 7))
    return rect.y + HDR


def card_edge(rect):
    rrect(screen, EDGE, rect, r=12, width=1)


def picture(img, rect, crop_y=0):
    """a picture inside a card, bottom corners rounded to match the card"""
    screen.blit(img, rect.topleft, area=pygame.Rect(0, crop_y, rect.w, rect.h))
    screen.blit(CORNER_L, (rect.x, rect.bottom - 12)); screen.blit(CORNER_R, (rect.right - 12, rect.bottom - 12))


def pill(right_x, y, text, fg, bg, dot=None):
    """a rounded label ending at right_x; returns the x where the next pill (to its left) must end"""
    t = small.render(text, True, fg); w = t.get_width() + (32 if dot else 20)
    rrect(screen, bg, (right_x - w, y, w, 26), r=13)
    if dot: pygame.draw.circle(screen, dot, (right_x - w + 14, y + 13), 4)
    screen.blit(t, (right_x - w + (24 if dot else 10), y + 4))
    return right_x - w - 8


def wrap(text, f, width):
    lines, line = [], ""
    for w_ in text.split():
        if f.size((line + " " + w_).strip())[0] > width and line: lines.append(line); line = w_
        else: line = (line + " " + w_).strip()
    if line: lines.append(line)
    return lines


def chip(x, y, key, label, col=(52, 56, 72)):
    """a keycap + label; returns the x after it"""
    k = mono.render(key, True, INK); w = k.get_width() + 12
    rrect(screen, (col[0] // 2, col[1] // 2, col[2] // 2), (x, y + 2, w, 20), r=5); rrect(screen, col, (x, y, w, 20), r=5)
    screen.blit(k, (x + 6, y + 1))
    l = small.render(label, True, DIM); screen.blit(l, (x + w + 5, y + 1))
    return x + w + 5 + l.get_width() + 12


def meter(x, y, w, hz, bar, col, top=None, h=10):
    """firing-rate bar with the pass bar marked"""
    top = top or max(bar * 2.5, hz * 1.15, 60.0)
    rrect(screen, (40, 43, 56), (x, y, w, h), r=h // 2)
    if hz > 0: rrect(screen, col, (x, y, max(int(w * min(hz / top, 1.0)), h), h), r=h // 2)
    bx = x + int(w * min(bar / top, 1.0)); pygame.draw.line(screen, INK, (bx, y - 4), (bx, y + h + 3), 2)
clock = pygame.time.Clock()
worker = BrainWorker(args.sex)
body = FlyBody(camera=args.camera, sugar_at=(1e6, 1e6), bitter_at=(1e6, 1e6), water_at=(1e6, 1e6), buffer=False,   # drops parked off-screen
               ctrl_every=10)            # leg controller at 1 kHz (physics still at 10 kHz): ~4.7x faster, same walking
geoms = {"sugar": body.sim.mj_model.geom("sugar"), "bitter": body.sim.mj_model.geom("bitter"), "water": body.sim.mj_model.geom("water")}

# which behaviour the red/blue team acts on, given what the fly is doing right now: every stimulus maps to its own
# behaviour.
TARGET = {"sugar": "eat", "water": "drink", "bitter": "refuse_bitter", "looming": "escape",
          "approach": "retreat", "threat": "turn", "hijack": "eat"}

QUICK_TASTE = {"eat": "sugar", "drink": "water", "refuse_bitter": "bitter"}

LABELS = {"eat": ("sugar taste neurons", "MN9 (feeding muscle)"), "drink": ("water taste neurons", "MN9 (feeding muscle)"),
          "refuse_bitter": ("sugar + bitter taste neurons", "MN9 (feeding muscle)"), "escape": ("looming detectors LC4/LPLC2", "giant fiber DNp01 (escape)"),
          "retreat": ("approach detectors LC16, both eyes", "MDN moonwalker (walk backward)"), "turn": ("looming detectors, one eye", "DNa02 (steering)")}

state = dict(walking=False, steer=0.0, target=None, kind=None, tour=[], spots={}, explore=False, wander=0.0, drops={}, meter=None, phase="idle", t_phase=0.0, eat=0.0, turn_dir=0.0, verdict=None, verdict_col=(255, 255, 255),
             msg="loading the brain ...", loglines=[], last_frame=None, brain_img=None, anim=None,
             silenced=None, counts=None,
             drag=None, spin=False, t_spin=None,
             explain=("", "", ""), score=None, show_log=False,
             thumbs=None, t_healthy=False, t_att=False, t_rep=False, lost=False)   # stepper: which tests have been done   # before / after pictures: dict(name, healthy=(img, hz), attack=(img, hz), repair=(img, hz) | "lost")   # explanation panel lines, red/blue scoreboard, TAB = technical log           # 3-D brain view: mouse-drag origin, auto-spin on/off   # counts: dict(fired, missing, silenced, stage) for the brain-panel line


def place_drop(kind, dist=9.0):
    pos = body.position[:2]; h = float(body.heading)
    tgt = pos + dist * np.array([np.cos(h), np.sin(h)])
    for k_, g in geoms.items(): body.sim.mj_model.geom_pos[g.id][:2] = tgt if k_ == kind else (1e6, 1e6)
    state["drops"] = {kind: tgt}
    state.update(target=tgt, kind=kind, walking=True, phase="approach", verdict=None)
    state["msg"] = f"{kind} placed {dist:.0f} mm ahead; walking to it" if dist >= ARRIVE else f"quick test: {kind} placed at the mouth"


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


THUMB = (140, 60)                                # before / after pictures in the scoreboard card


def draw_score(st):
    """SCOREBOARD card: healthy / attacked / repaired brain pictures side by side, each with its readout bar.
    Before any attack (and in Phase 2) it shows the readout of the last test against its pass bar."""
    r = SCORE_CARD; th = st["thumbs"] if args.mode != "phase2" else None
    if not th:
        y = card(r, "Readout" if args.mode == "phase2" else "Scoreboard", accent=GREEN if args.mode == "phase2" else None)
        if st["meter"]:
            label, hz, bar_, ok = st["meter"]
            t = title_font.render(f"{hz:.0f} Hz", True, GREEN if ok else RED); screen.blit(t, (r.x + 16, y + 6))
            screen.blit(font.render(label, True, INK), (r.x + 26 + t.get_width(), y + 12))
            meter(r.x + 16, y + 48, r.w - 32, hz, bar_, GREEN if ok else RED, h=14)
            screen.blit(small.render(f"white line = the pass bar ({bar_:.0f} Hz): above it, this neuron counts as firing", True, DIM), (r.x + 16, y + 72))
        else:
            t = font.render("No test yet: pick a behaviour" + (" above." if args.mode == "phase2" else " and test the healthy fly."), True, DIM)
            screen.blit(t, (r.x + 16, y + 12))
        if args.mode != "phase2":
            hint = "Press A to send in the RED TEAM (F: hijack attack)" if worker.attack is None else ""
            if hint: screen.blit(font.render(hint, True, (255, 150, 140)), (r.x + 16, r.bottom - 34))
        card_edge(r); return
    name = th["name"]; short = SHORT.get(name, "output"); healthy = th["healthy"][1]
    right = f"red {worker.attack['size'] if worker.attack else '-'}  vs  blue " + ("lost" if th.get("repair") == "lost" else (str(worker.repair["size"]) if worker.repair else "-"))
    y = card(r, f"Scoreboard: {NAMES_UI.get(name, name)}", right=f"neurons used: {right}")
    vals = [th[k][1] for k in ("healthy", "attack", "repair") if k in th and th[k] != "lost"]
    bar_ = worker.bhv[name].on_thr; top = max(bar_ * 2.0, max(vals) * 1.12, 20.0)
    x = r.x + 14; cw = THUMB[0]
    for key, lab, col in (("healthy", "HEALTHY" + (" (no food)" if not th["with_stim"] else ""), (200, 205, 220)), ("attack", "ATTACKED", RED), ("repair", "REPAIRED", BLUE)):
        has = key in th; lost = has and th[key] == "lost"
        screen.blit(cap.render(lab, True, col), (x, y + 6))
        box = pygame.Rect(x, y + 24, *THUMB)
        if has and not lost:
            screen.blit(pygame.surfarray.make_surface(np.transpose(th[key][0], (1, 0, 2))), box.topleft)
            rrect(screen, col, box.inflate(4, 4), r=6, width=2)
            hz = th[key][1]; meter(x, box.bottom + 12, cw, hz, bar_, col, top=top, h=8)
            pct = f"  {hz / healthy * 100:.0f}%" if healthy > 1 and th["with_stim"] and name != "refuse_bitter" else ""
            screen.blit(small.render(f"{short} {hz:.0f} Hz{pct}", True, INK), (x, box.bottom + 24))
        else:
            rrect(screen, (18, 19, 25), box, r=6); rrect(screen, col if lost else EDGE, box.inflate(4, 4), r=6, width=2 if lost else 1)
            t = small.render("no repair works" if lost else "press D to repair" if key == "repair" else "", True, col if lost else DIM)
            screen.blit(t, (box.centerx - t.get_width() // 2, box.centery - 9))
        x += cw + 16
    card_edge(r)


NAMES_UI = {"eat": "eating", "drink": "drinking", "refuse_bitter": "refusing bitter", "escape": "escaping",
            "retreat": "backing away", "turn": "turning away"}


def draw_strip(cur_name):
    """Phase 3: the five steps of the red/blue story, the current one lit. Phase 2: the six behaviours, the current one lit."""
    y = STRIP_Y; st = state
    if args.mode == "phase2":
        items = [("eat", "Eat", "S"), ("drink", "Drink", "V"), ("refuse_bitter", "Refuse bitter", "B"), ("escape", "Escape", "L"),
                 ("retreat", "Back away", "DOWN"), ("turn", "Turn away", "2")]
        w = (W - 24 - 5 * 10) // 6; x = 12
        for nm, lab, key in items:
            on = nm == cur_name and st["kind"] is not None
            rrect(screen, (40, 46, 36) if on else PANEL, (x, y, w, 32), r=16); rrect(screen, GREEN if on else EDGE, (x, y, w, 32), r=16, width=2 if on else 1)
            t = font.render(lab, True, INK if on else DIM); k = mono.render(key, True, INK)
            tw = t.get_width() + 10 + k.get_width() + 10; x0 = x + (w - tw) // 2
            screen.blit(t, (x0, y + 5)); rrect(screen, (60, 64, 80), (x0 + t.get_width() + 8, y + 7, k.get_width() + 10, 19), r=4); screen.blit(k, (x0 + t.get_width() + 13, y + 8))
            x += w + 10
        return
    lost = st["lost"]
    steps = [("Test the healthy fly", "", (200, 205, 220)), ("Red team attacks", "A", RED), ("Test it", "T", ACCENT),
             ("Blue team repairs" if not lost else "Blue team loses", "D", BLUE if not lost else (200, 90, 90)), ("Test it again", "T", ACCENT)]
    attacked = worker.attack is not None; repaired = worker.repair is not None or lost
    cur = (0 if not st["t_healthy"] else 1) if not attacked else ((2 if not st["t_att"] else 3) if not repaired else (4 if not st["t_rep"] else 5))
    w = (W - 24 - 4 * 22) // 5; x = 12; pulse = 0.5 + 0.5 * np.sin(time.time() * 3)
    for i, (lab, key, col) in enumerate(steps):
        done = i < cur and (i != 0 or st["t_healthy"]); now = i == cur
        bg = tuple(int(c * 0.22) for c in col) if done else ((32, 34, 44) if now else PANEL)
        rrect(screen, bg, (x, y, w, 32), r=16)
        edge = tuple(int(c * (0.65 + 0.35 * pulse)) for c in col) if now else (col if done else EDGE)
        rrect(screen, edge, (x, y, w, 32), r=16, width=2 if (now or done) else 1)
        pygame.draw.circle(screen, col if (done or now) else (60, 63, 78), (x + 18, y + 16), 10)
        if done and i == 3 and lost:                                                # the blue team lost: a cross, not a tick
            pygame.draw.line(screen, (14, 14, 18), (x + 13, y + 11), (x + 23, y + 21), 3); pygame.draw.line(screen, (14, 14, 18), (x + 23, y + 11), (x + 13, y + 21), 3)
        elif done: pygame.draw.lines(screen, (14, 14, 18), False, [(x + 13, y + 16), (x + 17, y + 20), (x + 24, y + 11)], 3)   # a tick
        else:
            n = cap.render(str(i + 1), True, (14, 14, 18) if now else DIM)
            screen.blit(n, (x + 18 - n.get_width() // 2, y + 16 - n.get_height() // 2))
        t = font.render(lab, True, INK if (now or done) else DIM); screen.blit(t, (x + 36, y + 5))
        if key:
            kk = mono.render(key, True, INK); kx = x + w - kk.get_width() - 20
            if kx > x + 36 + t.get_width() + 6:
                rrect(screen, (60, 64, 80), (kx - 5, y + 7, kk.get_width() + 10, 19), r=4); screen.blit(kk, (kx, y + 8))
        if i < 4:                                                                   # the arrow to the next step
            ax = x + w + 6; pygame.draw.polygon(screen, (80, 84, 100), [(ax, y + 10), (ax + 10, y + 16), (ax, y + 22)])
        x += w + 22


def draw_text(lines, x, y, col=(230, 230, 230)):
    for i, l in enumerate(lines): screen.blit(font.render(l, True, col), (x, y + 18 * i))


running = True; escape_kick = 0.0; t_loaded = None; frames_out = []; t_start = time.time()
REC_FPS = 10; rec_n = 0                                     # the video plays in real time: frames are repeated / skipped by the clock
if args.record:
    import imageio.v2 as iio; rec_w = iio.get_writer(args.record, fps=REC_FPS, codec="libx264", quality=8, macro_block_size=8)
script = [(float(x.split("@")[1]), x.split("@")[0].strip().upper()) for x in args.script.split(",") if "@" in x]
while running:
    # scripted keys: each fires N seconds after the system went idle (brain done, fly not busy) following the previous key
    idle = worker.brain is not None and worker.q.empty() and worker.busy == "" and state["phase"] in ("idle", "free", "explore") and state["anim"] is None
    if args.max_seconds and time.time() - t_start > args.max_seconds: running = False       # safety: a script can never hang
    if idle and t_loaded is None: t_loaded = time.time()
    if not idle: t_loaded = None
    if t_loaded is not None and script and time.time() - t_loaded >= script[0][0]:
        _, key = script.pop(0); t_loaded = None
        if os.environ.get("LIVE_PROFILE"): print(f"script: pressing {key} at t={time.time()-t_start:.0f}s phase={state['phase']} msg={state['msg']!r}", flush=True)
        if key in ("LCLICK", "RCLICK", "MCLICK"):                    # scripted mouse clicks on the fly panel (for tests / clips)
            btn = {"LCLICK": 1, "RCLICK": 3, "MCLICK": 2}[key]; x = FLY_C.x + {"LCLICK": int(PW * 0.25), "RCLICK": int(PW * 0.75), "MCLICK": PW // 2}[key]
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=btn, pos=(x, FLY_C.centery), scripted=True))
        else:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{key.lower()}") if len(key) == 1 else getattr(pygame, f"K_{key}")))
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running = False
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:                       # full screen <-> window
            FULL = not FULL; display = open_display(FULL)
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            log_event("key", key=pygame.key.name(k), mode=args.mode, kind_before=state["kind"])
            if k == pygame.K_q: running = False
            elif worker.brain is None: pass
            # T = quick test: the same brain test as the behaviour's own key, on whatever was attacked (or is on screen),
            # with the drop put straight at the mouth -- no walk. The taste is judged by the same arrival code as S / V / B.
            elif k == pygame.K_t:
                name_ = worker.attack_name if (worker.attack is not None and worker.attack_name) else TARGET.get(state["kind"], "eat")
                if worker.attack is not None and worker.attack_goal == "force" and args.mode != "phase2":
                    k = pygame.K_h
                elif name_ in QUICK_TASTE:
                    place_drop(QUICK_TASTE[name_], dist=0.3); k = None
                else:
                    k = {"escape": pygame.K_l, "retreat": pygame.K_DOWN, "turn": pygame.K_2}[name_]
            if k is None or ev.key == pygame.K_q or worker.brain is None: pass
            elif k == pygame.K_s: place_drop("sugar")
            elif k == pygame.K_b: place_drop("bitter")
            elif k == pygame.K_v: place_drop("water")
            elif k == pygame.K_c: place_arena()
            elif k == pygame.K_e:
                state["explore"] = not state["explore"]
                if state["explore"]: state.update(phase="explore", walking=True, target=None, tour=[], verdict=None); state["msg"] = "EXPLORE: the fly wanders; click to place drops (left = sugar, right = bitter, middle = water)"
                else: state.update(phase="idle", walking=False); state["msg"] = "explore off"
            elif k == pygame.K_w: state["walking"] = not state["walking"]; state["phase"] = "free"; state["msg"] = "walking" if state["walking"] else "stopped"
            elif k == pygame.K_LEFT: state["steer"] = min(state["steer"] + 0.4, 0.8)
            elif k == pygame.K_RIGHT: state["steer"] = max(state["steer"] - 0.4, -0.8)
            elif k == pygame.K_0 and worker.brain is not None:
                worker.view.reset_view(); state["spin"] = False; state["msg"] = "brain view: front"
            elif k == pygame.K_o and worker.brain is not None:
                state["spin"] = not state["spin"]; state["t_spin"] = None
                state["msg"] = "brain view: slowly spinning (O to stop, 0 for the front view)" if state["spin"] else "brain view: spin stopped"
            elif k == pygame.K_SPACE: state.update(walking=False, steer=0.0, target=None, phase="idle", verdict=None, tour=[], explore=False); state["msg"] = "stopped"
            elif k == pygame.K_l: state.update(phase="think", kind="looming", walking=False); state["msg"] = "looming shadow! brain: escape?"; worker.q.put(("measure", "escape", True))
            elif k == pygame.K_DOWN: state.update(phase="think", kind="approach", walking=False); state["msg"] = "object approaching head-on (LC16, both eyes): brain: back up?"; worker.q.put(("measure", "retreat", True))
            elif k in (pygame.K_1, pygame.K_2):
                side = "L" if k == pygame.K_1 else "R"; state.update(phase="think", kind="threat", walking=False)
                state["msg"] = f"threat on the {'LEFT' if side == 'L' else 'RIGHT'} eye: brain: which steering neuron fires?"; worker.q.put(("threat", side))
            # ---- PHASE 3 KEYS ONLY BELOW THIS LINE -------------------------------------------------------------
            # Phase 2 is the behaviour demo and must never attack or repair anything: every red/blue key is swallowed
            # here, BEFORE any of them is handled.
            elif args.mode == "phase2" and k in (pygame.K_a, pygame.K_f, pygame.K_h, pygame.K_d, pygame.K_r):
                state["msg"] = "attack / repair belong to Phase 3 -- run:  python virtual_fly/phase3.py"
            elif k == pygame.K_h: state.update(phase="think", kind="hijack", walking=False); state["msg"] = "HIJACK: no sugar at all; does the attacked/forced brain eat?"; worker.q.put(("measure", "eat", False))
            elif k == pygame.K_a:
                name = TARGET.get(state["kind"], "eat")            # whatever the fly is doing right now
                goal = "corrupt" if BEHAVIOURS[worker.sex][name].get("expected") == "off" else "block"
                state["msg"] = f"RED TEAM: attacking {name} ..."; worker.q.put(("attack", name, goal))
            elif k == pygame.K_f:
                state["msg"] = "RED TEAM: hijack -- make the fly eat with NO sugar (goal force) ..."; state["kind"] = "sugar"; worker.q.put(("attack", "eat", "force"))
            elif k == pygame.K_d and worker.attack is not None and worker.repair is not None:
                state["msg"] = "BLUE TEAM: already repaired. Test it (S/B/V/L/DOWN/2), attack again (A), or reset (R)."
            elif k == pygame.K_d and worker.attack is None:
                # nothing is broken: there is nothing for the defender to repair
                state["msg"] = "BLUE TEAM: nothing to repair -- the brain is healthy. Attack it first (A or F)."
            elif k == pygame.K_d:
                name = worker.attack_name or TARGET.get(state["kind"], "eat")   # repair what was actually attacked
                # ...and against the goal it was actually attacked with: after F (force) the damage is "fires with NO
                # stimulus", which is a different repair from "silent WITH the stimulus" (block)
                goal = worker.attack_goal or ("corrupt" if BEHAVIOURS[worker.sex][name].get("expected") == "off" else "block")
                state["msg"] = f"BLUE TEAM: repairing {name} ..."; worker.q.put(("repair", name, goal))
            elif k == pygame.K_r: worker.q.put(("reset",)); state["msg"] = "brain reset: no attacks, no repairs"
            elif k == pygame.K_m:
                worker = BrainWorker("male" if worker.sex == "female" else "female"); state["msg"] = f"loading the {worker.sex} brain ..."
            if k == pygame.K_TAB: state["show_log"] = not state["show_log"]
            ex_ = explain_key(pygame.key.name(ev.key), args.mode, worker.attack is not None, worker.repair is not None)
            if any(ex_): state["explain"] = ex_
            log_event("after_key", key=pygame.key.name(ev.key), msg=state["msg"], explain=state["explain"])     # what the screen says right after the key

        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and worker.brain is not None and BRAIN_C.collidepoint(to_canvas(ev)):
            state["drag"] = to_canvas(ev)
        elif ev.type == pygame.MOUSEMOTION and state["drag"] is not None and worker.brain is not None:
            mp = to_canvas(ev); dx, dy = mp[0] - state["drag"][0], mp[1] - state["drag"][1]; state["drag"] = mp
            worker.view.rotate_by(dx * 0.01, dy * 0.01)                  # about 0.6 degrees per pixel dragged
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            state["drag"] = None
        elif ev.type == pygame.MOUSEWHEEL and worker.brain is not None and BRAIN_C.collidepoint(to_canvas(pygame.mouse.get_pos())):
            worker.view.zoom_by(1.15 ** ev.y)
        elif ev.type == pygame.MOUSEBUTTONDOWN and worker.brain is not None and FLY_C.collidepoint(to_canvas(ev)):
            kind = {1: "sugar", 3: "bitter", 2: "water"}.get(ev.button)
            if kind:
                angle = (FLY_C.centerx - to_canvas(ev)[0]) / (PW / 2) * 1.0   # click left of centre -> to the fly's left (up to ~57 deg)
                drop_at(kind, angle)
                if not state["explore"]: state.update(target=state["drops"][kind], kind=kind, walking=True, phase="approach", verdict=None)

    # ---- brain results
    try:
        while True:
            kind, data = worker.out.get_nowait()
            if kind == "loaded":
                state["msg"] = (f"{worker.sex.capitalize()} brain ready: {worker.brain.N:,} neurons. Pick a behaviour above to test it."
                                if args.mode == "phase2" else f"{worker.sex.capitalize()} brain ready. Test the healthy fly first, then press A to attack.")
            elif kind == "measured":
                d = data; st = state
                if d["with_stim"] or st["kind"] == "hijack":        # draw the stimulus run; the no-stimulus control is only logged
                    st["anim"] = dict(t0=time.time(), spike_t=d["spike_t"], spike_i=d["spike_i"], dur=3.0, fresh=True)
                    st["counts"] = run_counts(d["spike_i"]); st["readout_hz"] = (d["name"], d["hz"])
                    if not worker.iv: st["silenced"] = None
                st["loglines"].append(f"brain run: {d['name']} {'WITH' if d['with_stim'] else 'without'} stimulus -> readout {d['hz']:.0f} Hz")
                if d["name"] == "escape" and d["with_stim"]:
                    ok = d["hz"] >= d["on_thr"]; st["verdict"] = "ESCAPES" if ok else "NO ESCAPE"; st["verdict_col"] = (120, 230, 120) if ok else (240, 110, 110)
                    st["phase"] = "escape" if ok else "idle"; escape_kick = 1.0 if ok else 0.0; st["t_phase"] = 0.0
                    st["msg"] = f"giant fiber DNp01 at {d['hz']:.0f} Hz (bar {d['on_thr']:.0f}) -> {st['verdict']}"; st["meter"] = ("giant fiber DNp01", d["hz"], d["on_thr"], ok)
                elif d["name"] == "retreat" and d["with_stim"]:
                    ok = d["hz"] >= d["on_thr"]; st["verdict"] = "BACKS UP" if ok else "NO RETREAT"; st["verdict_col"] = (120, 230, 120) if ok else (240, 110, 110)
                    st["phase"] = "retreat" if ok else "idle"; st["t_phase"] = 0.0
                    st["msg"] = f"moonwalker neurons MDN at {d['hz']:.0f} Hz (bar {d['on_thr']:.0f}) -> {st['verdict']}"; st["meter"] = ("moonwalker MDN", d["hz"], d["on_thr"], ok)
                elif d["name"] == "eat" and not d["with_stim"] and st["kind"] == "hijack":
                    ok = d["hz"] >= d["on_thr"]; st["verdict"] = "EATS NOTHING" if ok else "no feeding"; st["verdict_col"] = (240, 200, 90) if ok else (200, 200, 200)
                    st["phase"] = "eating" if ok else "idle"; st["t_phase"] = 0.0
                    st["msg"] = f"MN9 with NO sugar: {d['hz']:.0f} Hz -> {st['verdict']}"; st["meter"] = ("MN9 with no sugar", d["hz"], d["on_thr"], ok)
                elif d["with_stim"] and st["kind"] in ("sugar", "bitter", "water"):
                    ok = d["hz"] >= d["on_thr"]
                    if st["kind"] == "sugar": st["verdict"] = "EATS" if ok else "NO FEEDING"; st["verdict_col"] = (120, 230, 120) if ok else (240, 110, 110)
                    elif st["kind"] == "water": st["verdict"] = "DRINKS" if ok else "NO DRINKING"; st["verdict_col"] = (120, 200, 255) if ok else (240, 110, 110)
                    else:
                        # refusal is judged against the OFF bar, not the ON bar: between the two the brain is neither
                        # eating nor cleanly refusing, so it is shown as partly corrupted.
                        off_bar = worker.bhv["refuse_bitter"].off_thr
                        if ok: st["verdict"] = "EATS BITTER!"; st["verdict_col"] = (240, 110, 110)
                        elif d["hz"] <= off_bar: st["verdict"] = "REFUSES BITTER"; st["verdict_col"] = (120, 230, 120)
                        else: st["verdict"] = "PARTLY CORRUPTED"; st["verdict_col"] = (240, 200, 90)
                    st["phase"] = "eating" if ok else "refuse"; st["t_phase"] = 0.0
                    if st["kind"] == "bitter" and st["verdict"] == "PARTLY CORRUPTED": st["phase"] = "refuse"
                    st["msg"] = f"MN9 {d['hz']:.0f} Hz with {st['kind']} (bar {d['on_thr']:.0f}) -> {st['verdict']}"
                    st["meter"] = (f"MN9 with {st['kind']}", d["hz"], d["on_thr"], ok if st["kind"] != "bitter" else not ok)
                elif not d["with_stim"]:
                    st["loglines"].append(f"   (control without stimulus: {d['hz']:.0f} Hz, must be <= {d['off_thr']:.0f})")
            elif kind == "threat":
                d = data; st = state; st["anim"] = dict(t0=time.time(), spike_t=d["spike_t"], spike_i=d["spike_i"], dur=3.0, fresh=True)
                st["counts"] = run_counts(d["spike_i"]); st["readout_hz"] = ("turn", d["dna_L"])
                if not worker.iv: st["silenced"] = None
                diff = d["dna_L"] - d["dna_R"]                          # steering neurons fire asymmetrically -> a turn
                if abs(diff) >= d["on_thr"]:
                    # the BRAIN decides the direction: DNa02 drives ipsilateral turning, so whichever side fires harder
                    # is the way the fly turns, including INTO the threat if the brain has been corrupted.
                    turn = 1.0 if diff > 0 else -1.0                     # +1 = turn left
                    away = 1.0 if d["side"] == "R" else -1.0             # the direction that escapes this threat
                    ok = turn == away
                    st["verdict"] = ("TURNS LEFT" if turn > 0 else "TURNS RIGHT") + ("" if ok else " -- INTO THE THREAT")
                    st["verdict_col"] = (120, 230, 120) if ok else (240, 110, 110)
                    st["phase"] = "turn"; st["t_phase"] = 0.0; st["turn_dir"] = turn
                else: st["verdict"] = "NO TURN"; st["verdict_col"] = (240, 110, 110); st["phase"] = "idle"
                st["loglines"].append(f"brain run: threat on {d['side']} eye -> DNa02 left {d['dna_L']:.0f} Hz, right {d['dna_R']:.0f} Hz")
                st["msg"] = f"steering neurons: DNa02 L {d['dna_L']:.0f} Hz, R {d['dna_R']:.0f} Hz (bar {d['on_thr']:.0f}) -> {st['verdict']}"
                st["meter"] = ("DNa02 left minus right", abs(diff), d["on_thr"], abs(diff) >= d["on_thr"])
            elif kind == "attacked":
                b_ = data["best"]
                for row in data["log"][-6:]: state["loglines"].append(f"  [red {row['step']:2d}] {'GOAL ' if row['goal_met'] else '     '}{row['hypothesis'][:48]:48s} -> {row['result']}")
                tk_ = {"eat": "S", "drink": "V", "refuse_bitter": "B", "escape": "L", "retreat": "DOWN", "turn": "2"}.get(worker.attack_name, "S")
                if data.get("goal") == "force": tk_ = "H"
                state["msg"] = (f"RED TEAM: {data['goal']} with {b_['size']} neurons ({b_['strategy'].replace('attacker-', '')}). Press {tk_} to test the fly."
                                if b_ else "RED TEAM: no attack found within budget")
            elif kind == "repaired":
                r_ = data["rep"]
                for row in data["log"][-6:]: state["loglines"].append(f"  [blue {row['step']:2d}] {'GOAL ' if row['goal_met'] else '     '}{row['hypothesis'][:48]:48s} -> {row['result']}")
                test_key = {"eat": "S", "drink": "V", "refuse_bitter": "B", "escape": "L", "retreat": "DOWN", "turn": "2"}.get(worker.attack_name, "S")
                if not r_:
                    state["msg"] = "BLUE TEAM: no repair found within budget"
                elif r_.get("partial"):
                    state["msg"] = "BLUE TEAM LOST: " + r_.get("why", "no repair restores this behaviour")
                else:
                    rec = ""
                    state["msg"] = (f"BLUE TEAM: repaired with {r_['size']} neurons ({r_['move']}){rec}"
                                    f"{' [fragile]' if r_.get('fragile') else ''}. Press {test_key} to test.")
            elif kind == "replay":
                d = data; st = state
                st["anim"] = dict(t0=time.time(), spike_t=d["spike_t"], spike_i=d["spike_i"], dur=3.0, fresh=True,
                                  snap=f"{d['name']}_after_{d['stage']}")
                st["silenced"] = d["silenced"]; st["silenced_for"] = d["name"]; st["counts"] = run_counts(d["spike_i"])
                st["readout_hz"] = (d["name"], d["hz"])
                st["verdict"] = None                                   # the old badge described the brain BEFORE this change
                hz_, on_, off_ = d["hz"], d["on_thr"], d["off_thr"]
                normal_ = (hz_ <= off_) if (d["name"] == "refuse_bitter" or not d["with_stim"]) else (hz_ >= on_)
                st["meter"] = (f"{LABELS[d['name']][1].split(' (')[0]} after the {d['stage']}", hz_, on_, normal_)
                if args.mode != "phase2":
                    b_ = worker.bhv[d["name"]]; stim_ = b_.stim if d["with_stim"] else ()
                    att_ = neurons_of(worker.attack["intervention"]) if worker.attack else []
                    rep_ = neurons_of(worker.repair["intervention"]) if (worker.repair and d["stage"] == "repair") else []
                    snap_ = lambda t_, i_, a_, r_: worker.view.snapshot(t_, i_, THUMB, stimulus=stim_, readout=b_.readout, readouts=b_.readouts,
                                                                      attacked=a_, repaired=r_)
                    if d["stage"] == "attack":
                        st["thumbs"] = dict(name=d["name"], with_stim=d["with_stim"], healthy=(snap_(d["h_spike_t"], d["h_spike_i"], [], []), d["healthy"]))
                    if st["thumbs"] and st["thumbs"]["name"] == d["name"]:
                        st["thumbs"][d["stage"]] = (snap_(d["spike_t"], d["spike_i"], att_, rep_), d["hz"])
                st["loglines"].append(f"brain after the {d['stage']}: readout {d['hz']:.0f} Hz, {d['n_fired']:,} neurons fired, "
                                      f"{d['n_silenced']:,} went silent")
            elif kind == "reset": state["msg"] = "brain reset"; state["thumbs"] = None; state.update(t_healthy=False, t_att=False, t_rep=False, lost=False); state["silenced"] = None; state["counts"] = None; state["readout_hz"] = None
            elif kind == "error": state["msg"] = "error: " + data
            state["loglines"] = state["loglines"][-7:]
            d_x = data if isinstance(data, dict) else {}
            nm_ = ({"measured": d_x.get("name"), "replay": d_x.get("name"), "threat": "turn"}.get(kind) or worker.attack_name)
            drawn_ = kind in ("threat", "replay", "attacked", "repaired", "reset") or (kind == "measured" and (d_x.get("with_stim") or state["kind"] == "hijack"))
            if nm_ in getattr(worker, "bhv", {}) and drawn_:
                bh_ = worker.bhv[nm_]
                ctx_ = dict(name=nm_, verdict=state.get("verdict"), attacked=worker.attack is not None,
                            healthy=worker.healthy_hz.get((nm_, worker.attack_goal != "force")), stim_count=len(bh_.stim),
                            fired=(state["counts"] or {}).get("fired"), on_thr=bh_.on_thr, off_thr=bh_.off_thr)
                ex_ = explain_result(kind, d_x, ctx_)
                if ex_: state["explain"] = tuple(n_ if n_ is not None else o_ for n_, o_ in zip(ex_, state["explain"]))
            if drawn_ and kind in ("measured", "threat"):                  # the stepper: a test of the brain as it now stands
                if worker.attack is None: state["t_healthy"] = True
                elif worker.repair is None and not state["lost"]: state["t_att"] = True
                else: state["t_rep"] = True
            if kind == "attacked": state.update(t_att=False, t_rep=False, lost=False)
            if kind == "repaired": state.update(t_rep=False, lost=bool(d_x.get("rep") and d_x["rep"].get("partial")))
            if kind == "replay":                                          # the scoreboard: what each team achieved
                if d_x["stage"] == "attack":
                    state["score"] = dict(name=d_x["name"], goal=worker.attack_goal, healthy=d_x["healthy"],
                                          red=(worker.attack["size"], d_x["hz"]), blue=None)
                elif state["score"]: state["score"]["blue"] = (worker.repair["size"], d_x["hz"])
            elif kind == "repaired" and d_x.get("rep") and d_x["rep"].get("partial") and state["score"]:
                state["score"]["blue"] = "lost"
                if state["thumbs"]: state["thumbs"]["repair"] = "lost"
            elif kind == "reset": state["score"] = None
            if args.events:                                   # what the screen now shows, and the brain it came from
                d_ = data if isinstance(data, dict) else {}
                pot = {f"{a_},{b_}": v for (a_, b_), v in worker.iv.get("potentiate", {}).items()} if worker.iv else {}
                log_event(kind, name=d_.get("name"), with_stim=d_.get("with_stim"), side=d_.get("side"),
                          stage=d_.get("stage"), n_fired=d_.get("n_fired"), n_silenced=d_.get("n_silenced"),
                          hz=d_.get("hz"), dna_L=d_.get("dna_L"), dna_R=d_.get("dna_R"),
                          verdict=state.get("verdict"), msg=state.get("msg"), meter=state.get("meter"),
                          attack_name=worker.attack_name, attack_goal=worker.attack_goal,
                          attack_size=(worker.attack or {}).get("size"), repair_size=(worker.repair or {}).get("size"),
                          explain=state["explain"], score=[t for t, _ in score_lines(state["score"])],
                          iv={k_: v_ for k_, v_ in worker.iv.items() if k_ != "potentiate"} | ({"potentiate": pot} if pot else {}))
    except queue.Empty: pass

    _t0 = time.time()
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
            if dist < 5.0:                                             # close: direct approach (same control as the S / B keys)
                st.update(target=spot, kind=kind, phase="approach", walking=True)
                st["msg"] = f"found the {kind} drop; walking up to it"
            else:
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
    _t1 = time.time()

    # ---- brain panel
    bimg = None
    if worker.brain is not None:
        v = worker.view; anim = st["anim"]
        if st["spin"] and st["drag"] is None:
            now_ = time.time(); v.rotate_by(0.35 * (now_ - st["t_spin"]) if st["t_spin"] else 0.0); st["t_spin"] = now_   # ~20 deg/s
        if anim is not None:
            if anim.get("fresh"): v.reset(); anim["fresh"] = False
            frac = min((time.time() - anim["t0"]) / anim["dur"], 1.0); v.advance(300.0 * frac, anim["spike_t"], anim["spike_i"])
            if frac >= 1.0:
                if anim.get("snap"): st["snap_pending"] = anim["snap"]
                st["anim"] = None
        else: v.glow *= 0.95
        cur_name = {"looming": "escape", "bitter": "refuse_bitter", "approach": "retreat", "threat": "turn", "water": "drink"}.get(st["kind"], "eat")
        cur = worker.bhv[cur_name]
        stim_label, readout_label = LABELS[cur_name]
        att = neurons_of(worker.attack["intervention"]) if worker.attack else []
        rep = neurons_of(worker.repair["intervention"]) if worker.repair else []
        sil = st["silenced"] if (st["silenced"] is not None and st.get("silenced_for") == cur_name and worker.iv) else ()
        c = st["counts"]; counts_txt = ""
        if c:
            counts_txt = f"{c['fired']:,} neurons fired" + (f" ({c['missing']:,} not drawn: no position)" if c["missing"] else "")
            off_ = int((v.fired & v.ok & ~v.vis).sum())
            if off_: counts_txt += f" ({off_:,} outside the zoomed view)"
            if args.mode != "phase2" and worker.iv:
                counts_txt += f"  |  {len(att)} attacked  |  {len(sil):,} went silent  |  {len(rep)} repaired"
        no_food = st["kind"] == "hijack"                                  # the no-food test: no taste neurons are stimulated
        if no_food: stim_label = stim_label + " (not stimulated: no food)"
        bimg = v.frame(stimulus=(() if no_food else cur.stim), readout=cur.readout, attacked=att, repaired=rep, stim_label=stim_label, readout_label=readout_label,
                       silenced=sil, counts=counts_txt, readouts=cur.readouts,
                       readout_hz=(st["readout_hz"][1] if st.get("readout_hz") and st["readout_hz"][0] == cur_name else None),
                       show_redblue=(args.mode != "phase2"), readout_bar=cur.on_thr,
                       title="", subtitle="", score=[],
                       stim_on=st["phase"] in ("think", "eating", "refuse", "escape", "retreat", "turn"), blink=0.5 + 0.5 * np.sin(time.time() * 4))

    if st.get("snap_pending") and bimg is not None:
        if args.snapshots:
            os.makedirs(args.snapshots, exist_ok=True)
            from PIL import Image as _Im; _Im.fromarray(bimg).save(os.path.join(args.snapshots, st["snap_pending"] + ".png"))
            st["snap_full"] = st["snap_pending"]                           # and the whole window, once it is drawn
        st["snap_pending"] = None
    _t2 = time.time()
    # ---- draw
    screen.fill(BG)
    # header: mark, title, status pills
    pygame.draw.circle(screen, RED, (30, 27), 13); pygame.draw.circle(screen, BLUE, (30, 27), 13, draw_top_right=True, draw_bottom_right=True)
    pygame.draw.circle(screen, BG, (30, 27), 5)
    screen.blit(title_font.render("NeuroRedTeam", True, INK), (52, 11))
    sub = ("Phase 2  \u00b7  a virtual fly run by its real connectome" if args.mode == "phase2"
           else "Phase 3  \u00b7  red team vs blue team on a real fly brain")
    screen.blit(font.render(sub, True, DIM), (52 + title_font.size("NeuroRedTeam")[0] + 14, 16))
    busy = worker.busy or st["phase"] == "think"
    status = worker.busy or ("brain thinking ..." if st["phase"] == "think" else "brain idle")
    xr = pill(W - 12, 13, status, INK, (70, 56, 24) if busy else (26, 52, 38), dot=(255, 190, 60) if busy else GREEN)
    if worker.brain is not None:
        if args.mode != "phase2":
            if not worker.iv: xr = pill(xr, 13, "healthy brain", INK, (30, 34, 46), dot=GREEN)
            else:
                txt = f"modified: attack {worker.attack['size']}" + (f"  +  repair {worker.repair['size']}" if worker.repair else "") + " neurons"
                xr = pill(xr, 13, txt, INK, (58, 30, 32), dot=RED if not worker.repair else BLUE)
        xr = pill(xr, 13, f"{worker.sex} fly  \u00b7  {worker.brain.N:,} neurons", DIM, (30, 34, 46))
    draw_strip(cur_name if worker.brain is not None else None)
    # the fly
    card(FLY_CARD, "The fly", right="FlyGym body  \u00b7  MuJoCo physics", accent=ACCENT)
    surf = pygame.surfarray.make_surface(np.transpose(fly_img, (1, 0, 2))); surf = pygame.transform.smoothscale(surf, (PW, FLY_H))
    picture(surf, FLY_C, crop_y=(FLY_H - PH) // 2)
    if st["verdict"]:
        t = big.render(st["verdict"], True, (12, 12, 16)); w_ = t.get_width() + 32; bx = FLY_C.right - w_ - 14
        rrect(screen, (0, 0, 0), (bx + 2, FLY_C.y + 16, w_, t.get_height() + 10), r=12)
        rrect(screen, st["verdict_col"], (bx, FLY_C.y + 12, w_, t.get_height() + 10), r=12); screen.blit(t, (bx + 16, FLY_C.y + 16))
    card_edge(FLY_CARD)
    # the brain
    card(BRAIN_CARD, "Its brain", right=("drag to turn  \u00b7  wheel to zoom" if worker.brain is not None else ""), accent=(120, 200, 255))
    if bimg is not None: picture(pygame.surfarray.make_surface(np.transpose(bimg, (1, 0, 2))), BRAIN_C)
    else:
        rrect(screen, (18, 19, 25), BRAIN_C, r=0)
        dots = "." * (1 + int(time.time() * 2) % 3)
        t = head.render(f"loading the {worker.sex} brain{dots}", True, DIM); screen.blit(t, (BRAIN_C.centerx - t.get_width() // 2, BRAIN_C.centery - 20))
        t = small.render("female ~10 s, male ~40 s", True, DIM); screen.blit(t, (BRAIN_C.centerx - t.get_width() // 2, BRAIN_C.centery + 10))
    card_edge(BRAIN_CARD)
    # what just happened
    r = STORY_CARD; y = card(r, "What just happened", right="TAB: technical log", accent=ACCENT)
    msg_ = st["msg"]; hf = next((f_ for f_ in HEAD_FONTS if f_.size(msg_)[0] <= r.w - 32), HEAD_FONTS[-1])   # shrink to fit first
    while msg_ and hf.size(msg_)[0] > r.w - 32: msg_ = msg_[:-2]
    screen.blit(hf.render(msg_ if msg_ == st["msg"] else msg_.rstrip() + " ...", True, ACCENT), (r.x + 16, y + 2 + (head.get_height() - hf.get_height()) // 2))
    yy = y + 30; y_max = r.bottom - 16
    if st["show_log"]:
        for line in st["loglines"][-6:]:
            if yy < y_max: screen.blit(mono.render(line[:96], True, (185, 187, 200)), (r.x + 16, yy)); yy += 19
    else:
        for lab, txt, col in zip(("You did", "The brain did", "What it means"), st["explain"], ((215, 218, 230), (255, 217, 89), (140, 230, 160))):
            if not txt: continue
            screen.blit(cap.render(lab.upper(), True, (120, 124, 142)), (r.x + 16, yy + 3))
            y0_ = yy
            for line in wrap(txt, body_f, r.w - 136 - 16):
                if yy < y_max: screen.blit(body_f.render(line[:1].upper() + line[1:] if yy == y0_ else line, True, col), (r.x + 136, yy))
                yy += 19
            yy += 3
    card_edge(r)
    draw_score(st)
    # keys, in families
    rows = [[("TEST", [("T", "quick test"), ("S", "sugar"), ("V", "water"), ("B", "bitter"), ("L", "looming"), ("DOWN", "approach"), ("1 2", "threat L / R")]
                      + ([("H", "no food")] if args.mode != "phase2" else []))]
            + ([("TEAMS", [("A", "attack"), ("F", "hijack"), ("D", "repair"), ("R", "reset")])] if args.mode != "phase2" else []),
            [("FLY", [("W", "walk"), ("\u2190 \u2192", "steer"), ("SPACE", "stop"), ("C", "choice arena"), ("E", "explore")]),
             ("VIEW", [("O", "spin"), ("0", "front"), ("TAB", "log"), ("F11", "window")]), ("", [("M", "male / female"), ("Q", "quit")])]]
    for ri, row in enumerate(rows):
        x = 16; y = KEYS_Y + 24 * ri
        for gname, keys in row:
            if gname:
                gcol = {"TEAMS": (255, 140, 130), "TEST": ACCENT}.get(gname, (150, 156, 176))
                g = cap.render(gname, True, gcol); screen.blit(g, (x, y + 3)); x += g.get_width() + 10
            for k_, lab in keys:
                col = (120, 44, 44) if k_ in ("A", "F") else (44, 64, 124) if k_ == "D" else (52, 56, 72)
                x = chip(x, y, k_, lab, col)
            x += 14
    present(); clock.tick(30)
    if st.get("snap_full") and args.snapshots:
        pygame.image.save(screen, os.path.join(args.snapshots, st["snap_full"] + "_window.png")); st["snap_full"] = None
    _t3 = time.time()
    if os.environ.get("LIVE_PROFILE"):
        _prof = state.setdefault("_prof", [0.0, 0.0, 0.0, 0]); _prof[0] += _t1 - _t0; _prof[1] += _t2 - _t1; _prof[2] += _t3 - _t2; _prof[3] += 1
        if _prof[3] % 20 == 0: print(f"per frame: body {_prof[0]/_prof[3]*1000:.0f} ms, brain panel {_prof[1]/_prof[3]*1000:.0f} ms, draw+flip {_prof[2]/_prof[3]*1000:.0f} ms, phase {st['phase']}", flush=True)
    if args.record:
        f_ = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))     # the canvas: always 1280 x 800, whatever the screen
        while rec_n < (time.time() - t_start) * REC_FPS: rec_w.append_data(f_); rec_n += 1     # straight to disk, never held in memory
pygame.quit()
if args.record: rec_w.close(); print("saved", args.record, rec_n, "frames")
