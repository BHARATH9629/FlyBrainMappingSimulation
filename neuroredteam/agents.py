"""
Red team (attacker) and blue team (defender), algorithmic versions.  v2: robust.

Shared loop: choose an experiment -> run the brain -> check the behaviour -> goal met on ALL check seeds? -> record,
then try to shrink the intervention set.  An LLM agent plugs into the same tools later; these are the baselines.

Goals
  block  : behaviour must disappear with the stimulus present         (readout <= off_thr)
  force  : behaviour must appear WITHOUT the stimulus                  (readout >= on_thr)
  corrupt: a stimulus the fly should refuse must trigger the behaviour (readout >= on_thr; behaviours with expected="off")
Attack moves (rules in behaviours.RULES): lesion, weaken, inject.  The readout neuron is never touched.
Defence moves (DEFENCE_RULES): bias, potentiate.  A repair must pass the two-condition test on all seeds.
"""
import numpy as np, time
from behaviours import RULES, DEFENCE_RULES

CHECK_SEEDS = (1, 2, 3)          # cheap screening: every candidate is tried on these seeds
ROBUST_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
# ROBUST_SEEDS: a result is only ACCEPTED -- and a neuron only dropped during minimisation -- if the goal holds on all of
# these, so that minimisation cannot stop at the edge of the threshold for a small set of seeds. Seeds 11 and up stay
# held out entirely (run_redteam.py, verify_phase3.py) as the independent check.


class Log:
    def __init__(self, quiet=False): self.rows = []; self.t0 = time.time(); self.quiet = quiet
    def add(self, agent, step, hypothesis, intervention, result, ok):
        self.rows.append(dict(agent=agent, step=step, hypothesis=hypothesis, intervention=intervention,
                              result=result, goal_met=ok, t=round(time.time() - self.t0, 1)))
        if not self.quiet:
            print(f"  [{agent} {step:2d}] {'GOAL' if ok else '    '} {hypothesis:<56s} {intervention:<34s} -> {result}")


def _fmt(iv):
    parts = []
    if iv.get("lesion"): parts.append(f"lesion {len(iv['lesion'])}")
    if iv.get("weaken"): parts.append(f"weaken {len(iv['weaken'])}")
    if iv.get("inject"): parts.append(f"inject {len(iv['inject'])}")
    if iv.get("bias"): parts.append(f"bias {len(iv['bias'])}")
    if iv.get("potentiate"): parts.append(f"potentiate {len(iv['potentiate'])}")
    return ", ".join(parts) or "nothing"


def merge_iv(*ivs):
    """Stack interventions: lesion lists union, dict moves union (later wins on conflict)."""
    out = {}
    for iv in ivs:
        for k, v in (iv or {}).items():
            if k == "lesion": out[k] = sorted(set(out.get(k, [])) | set(v))
            else: out[k] = {**out.get(k, {}), **v}
    return out


def size_of(iv):
    """Cost of an intervention = number of distinct neurons touched."""
    s = set(iv.get("lesion", [])) | set(iv.get("weaken", {})) | set(iv.get("inject", {})) | set(iv.get("bias", {}))
    s |= {a for (a, b) in iv.get("potentiate", {})}
    return len(s)


def readout_col(b):
    """Signed synapse counts from every neuron onto the WHOLE readout set. Behaviours with several readout neurons
    (retreat: all four MDNs) are judged on the mean over the set, so candidates are ranked against the set too."""
    return np.asarray(b.brain.W[:, b.readouts].sum(axis=1)).ravel()


def onto_readouts(b, S, f):
    """{(pre, readout): f} for every readout neuron each neuron in S actually synapses onto (potentiate / depress
    moves), so a repair acts on the whole readout set."""
    W = b.brain.W; out = {}
    for i in S:
        for r in ([r for r in b.readouts if W[i, r] != 0] or [b.readout]): out[(int(i), int(r))] = f
    return out


# ====================================================================== ATTACKER
class Attacker:
    def __init__(self, behaviour, log, goal="block", budget=RULES["budget"], seeds=CHECK_SEEDS, excluded=(), fixed=None):
        self.b, self.log, self.goal, self.budget, self.seeds = behaviour, log, goal, budget, seeds
        self.used = 0; self.excluded = set(excluded)      # neurons a previous attack used (for redundancy search)
        self.fixed = fixed or {}                          # interventions already on the brain (earlier attacks + repairs)
        self.excluded |= {int(i) for i in set(self.fixed.get("lesion", [])) | set(self.fixed.get("inject", {}))}   # do not re-use them

    # ---- the referee -------------------------------------------------------
    def _check(self, iv):
        """Goal met on every seed?  Returns (ok, mean readout Hz).
        block  : stimulus present, readout must be <= off_thr
        force  : no stimulus,      readout must be >= on_thr
        corrupt: stimulus present (a stimulus the fly should REFUSE), readout must be >= on_thr"""
        self.used += 1
        with_stim = self.goal in ("block", "corrupt")
        rates = [self.b.measure(with_stim, seed=s, **merge_iv(self.fixed, iv))[0] for s in self.seeds]
        ok = all(r <= self.b.off_thr for r in rates) if self.goal == "block" else all(r >= self.b.on_thr for r in rates)
        return ok, float(np.mean(rates))

    def _try(self, name, hypothesis, iv):
        ok, hz = self._check(iv)
        if ok: ok = self._confirm(iv)                     # screened on 3 seeds, accepted only if it holds on 8
        self.log.add(name, self.used, hypothesis, _fmt(iv), f"readout {hz:.1f} Hz", ok)
        return ok, hz

    def _confirm(self, iv):
        """The goal holds on every ROBUST_SEED (the screening seeds are already known to pass)."""
        extra = [s for s in ROBUST_SEEDS if s not in self.seeds]
        if not extra: return True
        self.used += 1
        with_stim = self.goal in ("block", "corrupt")
        rates = [self.b.measure(with_stim, seed=s, **merge_iv(self.fixed, iv))[0] for s in extra]
        return all(r <= self.b.off_thr for r in rates) if self.goal == "block" else all(r >= self.b.on_thr for r in rates)

    # ---- candidate generators ---------------------------------------------
    def drivers(self):
        """Excitatory inputs to the readout that are active with the stimulus, ranked by rate x synapses.
        If every direct driver is excluded (threat model T2), fall back one hop upstream: active neurons that excite
        the direct drivers, ranked by rate x (synapses onto drivers, weighted by the drivers' synapses onto the readout)."""
        _, r = self.b.measure(True, **self.fixed)
        W = self.b.brain.W; col = readout_col(self.b)
        score = {i: hz * col[i] for i, hz in r["active"].items()
                 if col[i] > 0 and i not in self.b.protected and i not in self.excluded}
        if not score and self.excluded:
            direct = [i for i, hz in r["active"].items() if col[i] > 0]
            if direct:
                wd = col[direct]                                              # drivers' weight onto the readout
                sub = W[:, direct]                                            # synapses onto the drivers
                for i, hz in r["active"].items():
                    if i in self.b.protected or i in self.excluded: continue
                    row = sub[i].toarray().ravel(); v = hz * float((row * (row > 0) * wd).sum())
                    if v > 0: score[int(i)] = v
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def brakes(self):
        """Inhibitory neurons active WITH the stimulus that project (directly or via one hop) onto the readout, strongest
        first: the candidates for a corrupt attack (cut the brake so a refused stimulus drives the behaviour)."""
        _, r = self.b.measure(True, **self.fixed)
        W = self.b.brain.W; ro = self.b.readout
        col = readout_col(self.b)
        exc = np.flatnonzero(col > 0)                              # direct excitatory inputs to the readout
        score = {}
        for i, hz in r["active"].items():
            if i in self.b.protected or i in self.excluded: continue
            direct = -col[i] if col[i] < 0 else 0.0
            row = W[i, exc].toarray().ravel(); indirect = -row[row < 0].sum() if row.size else 0.0
            v = hz * (direct + 0.5 * indirect)
            if v > 0: score[int(i)] = v
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])][:RULES["moves"]["inject"]["max_neurons"]]

    def inhibitors(self):
        """Neurons with inhibitory synapses onto the readout, strongest first (they are usually silent: inject wakes them).
        If all are excluded (T2), fall back to the strongest excitatory inputs OF those inhibitors (drive them indirectly)."""
        W = self.b.brain.W; col = readout_col(self.b)
        out = [int(i) for i in np.argsort(col) if col[i] < 0 and i not in self.b.protected and i not in self.excluded][:20]
        if not out and self.excluded:
            inh = [int(i) for i in np.argsort(col) if col[i] < 0][:20]
            sub = W[:, inh].toarray(); wi = -col[inh]
            v = (np.clip(sub, 0, None) * wi).sum(axis=1)
            out = [int(i) for i in np.argsort(-v) if v[i] > 0 and i not in self.b.protected and i not in self.excluded][:20]
        return out

    def excitatory_inputs(self):
        col = readout_col(self.b)
        return [int(i) for i in np.argsort(-col) if col[i] > 0 and i not in self.b.protected and i not in self.excluded][:20]

    # ---- shrink a successful set --------------------------------------------
    def _minimise(self, name, iv, key):
        items = list(iv[key]) if isinstance(iv[key], (list, tuple)) else list(iv[key].keys())
        for i in list(items):
            if self.used >= self.budget: break
            rest = [j for j in items if j != i]
            if not rest: continue
            trial = dict(iv); trial[key] = rest if isinstance(iv[key], (list, tuple)) else {j: iv[key][j] for j in rest}
            ok, _ = self._try(name, f"is neuron {i} necessary? try without it", trial)
            if ok: iv, items = trial, rest
        return iv

    # ---- strategies ---------------------------------------------------------
    def greedy(self, name="attacker-greedy"):
        """block: cut strongest drivers one at a time.  force: inject into strongest excitatory inputs."""
        if self.goal == "block":
            cands, key = self.drivers(), "lesion"
            make = lambda S: {"lesion": list(S)}
            cap = RULES["moves"]["lesion"]["max"]
        elif self.goal == "corrupt":
            cands, key = self.brakes(), "lesion"                  # cut the inhibitory gate that holds the readout down
            make = lambda S: {"lesion": list(S)}
            cap = RULES["moves"]["lesion"]["max"]
        else:
            cands, key = self.excitatory_inputs(), "inject"
            make = lambda S: {"inject": {i: RULES["moves"]["inject"]["max_hz"] for i in S}}
            cap = RULES["moves"]["inject"]["max_neurons"]
        S = []
        for i in cands[:cap]:
            if self.used >= self.budget: return None
            S.append(i); iv = make(S)
            ok, _ = self._try(name, f"{'cut' if key == 'lesion' else 'drive'} the {len(S)} strongest {'drivers' if key == 'lesion' else 'excitatory inputs'}", iv)
            if ok:
                iv = self._minimise(name, iv, key)
                return {"strategy": name, "goal": self.goal, "intervention": iv, "size": size_of(iv), "simulations": self.used}
        return None

    def inject_inhibitors(self, name="attacker-inject"):
        """block WITHOUT cutting anything: wake up the readout's own inhibitory inputs."""
        if self.goal != "block": return None
        cands = self.inhibitors(); hz = RULES["moves"]["inject"]["max_hz"]
        cap = RULES["moves"]["inject"]["max_neurons"]         # the ladder follows the limit in RULES
        for k in [k for k in (1, 2, 3, 5, 8, 12, 16, 20) if k <= cap]:
            if self.used >= self.budget or k > len(cands): return None
            iv = {"inject": {i: hz for i in cands[:k]}}
            ok, _ = self._try(name, f"drive the {k} strongest inhibitors of the readout (no cuts)", iv)
            if ok:
                iv = self._minimise(name, iv, "inject")
                return {"strategy": name, "goal": self.goal, "intervention": iv, "size": size_of(iv), "simulations": self.used}
        return None

    def weaken_only(self, name="attacker-weaken", factor=None):
        """block by only WEAKENING drivers (partial damage), never silencing them."""
        if self.goal != "block": return None
        f = factor or RULES["moves"]["weaken"]["min_factor"]; cands = self.drivers()
        for k in (1, 2, 3, 4, 6, 8, 10):
            if self.used >= self.budget or k > len(cands): return None
            iv = {"weaken": {i: f for i in cands[:k]}}
            ok, _ = self._try(name, f"weaken the {k} strongest drivers to {int(f*100)} %", iv)
            if ok:
                iv = self._minimise(name, iv, "weaken")
                return {"strategy": name, "goal": self.goal, "intervention": iv, "size": size_of(iv), "simulations": self.used}
        return None

    def random(self, name="attacker-random", rng=None):
        rng = rng or np.random.default_rng(0)
        _, r = self.b.measure(True, **self.fixed)
        pool = [i for i in r["active"] if i not in self.b.protected and i not in self.excluded]
        best = None
        while self.used < self.budget:
            k = int(rng.integers(1, RULES["moves"]["lesion"]["max"] + 1))
            iv = {"lesion": [int(i) for i in rng.choice(pool, k, replace=False)]}
            ok, _ = self._try(name, f"random guess of {k} active neurons", iv)
            if ok and (best is None or k < best["size"]):
                best = {"strategy": name, "goal": self.goal, "intervention": iv, "size": k, "simulations": self.used}
        return best

    def best_of_all(self, name="attacker-all"):
        """Run every strategy, return the smallest successful attack (ties: fewest simulations)."""
        found = [a for a in (self.greedy(), self.inject_inhibitors(), self.weaken_only()) if a]
        return min(found, key=lambda a: (a["size"], a["simulations"])) if found else None

    # ---- beam search: is greedy's answer the true minimum? -------------------
    def _make(self, moves):
        iv = {}
        for kind, i in moves:
            if kind == "lesion": iv.setdefault("lesion", []).append(int(i))
            elif kind == "inject": iv.setdefault("inject", {})[int(i)] = RULES["moves"]["inject"]["max_hz"]
            elif kind == "weaken": iv.setdefault("weaken", {})[int(i)] = RULES["moves"]["weaken"]["min_factor"]
        return iv

    def _minimise_moves(self, name, moves):
        """Drop moves one at a time while the goal still holds on all seeds."""
        moves = list(moves)
        for mv in list(moves):
            if self.used >= self.budget: break
            rest = [m for m in moves if m != mv]
            if not rest: continue
            ok, _ = self._try(name, f"is {mv[0]} of neuron {mv[1]} necessary? try without it", self._make(rest))
            if ok: moves = rest
        return moves

    def search(self, name="attacker-search", beam=4, pool=12, max_size=None, budget=None):
        """Beam search over mixed lesion / inject / weaken sets, seeded by the same candidate lists greedy uses.
        Scores each candidate set on ONE seed (cheap), verifies on all seeds only when the goal looks met, then
        minimises. Stops at max_size (default: the lesion cap) or when the budget is spent. Reports the smallest set."""
        budget = budget or RULES.get("search_budget", 150); max_size = max_size or RULES["moves"]["lesion"]["max"]
        if self.goal == "block":
            moves = [("lesion", i) for i in self.drivers()[:pool]] + [("inject", i) for i in self.inhibitors()[:pool // 2]] \
                  + [("weaken", i) for i in self.drivers()[:pool // 2]]
        elif self.goal == "corrupt":
            moves = [("lesion", i) for i in self.brakes()[:pool]] + [("inject", i) for i in self.excitatory_inputs()[:pool // 2]]
        else:
            moves = [("inject", i) for i in self.excitatory_inputs()[:pool]] + [("lesion", i) for i in self.inhibitors()[:pool // 2]]
        moves = [m for m in moves if m[1] not in self.protected_now()]
        with_stim = self.goal in ("block", "corrupt"); s0 = self.seeds[0]
        better = (lambda r: r) if self.goal == "block" else (lambda r: -r)          # lower is better
        met1 = (lambda r: r <= self.b.off_thr) if self.goal == "block" else (lambda r: r >= self.b.on_thr)
        states, seen, best = [()], set(), None
        for level in range(1, max_size + 1):
            children = []
            for st in states:
                for mv in moves:
                    if mv in st or any(m[1] == mv[1] for m in st): continue
                    child = tuple(sorted(st + (mv,)))
                    if child in seen: continue
                    seen.add(child)
                    if self.used >= budget: break
                    iv = self._make(child); self.used += 1
                    rate = self.b.measure(with_stim, seed=s0, **merge_iv(self.fixed, iv))[0]
                    hit = met1(rate)
                    self.log.add(name, self.used, f"beam level {level}: {', '.join(f'{k} {i}' for k, i in child)}", _fmt(iv), f"readout {rate:.1f} Hz (seed {s0})", hit)
                    if hit:
                        ok, hz = self._try(name, f"verify on all seeds", iv)
                        if ok:
                            mm = self._minimise_moves(name, child); ivm = self._make(mm)
                            cand = {"strategy": name, "goal": self.goal, "intervention": ivm, "size": size_of(ivm), "simulations": self.used}
                            if best is None or cand["size"] < best["size"]: best = cand
                    children.append((better(rate), child))
            if best is not None and best["size"] <= level: break      # nothing smaller can be found at deeper levels
            if not children or self.used >= budget: break
            children.sort(key=lambda c: c[0]); states = [c[1] for c in children[:beam]]
        return best

    def protected_now(self):
        return set(self.b.protected) | self.excluded


def redundancy(behaviour, log, goal="block", max_sets=4, budget=RULES["budget"]):
    """How many DISJOINT minimal attacks exist?  Find one, exclude its neurons, search again."""
    sets, excluded = [], set()
    for _ in range(max_sets):
        atk = Attacker(behaviour, log, goal=goal, budget=budget, excluded=excluded).greedy(name=f"redundancy-{len(sets)+1}")
        if atk is None: break
        neurons = set(atk["intervention"].get("lesion", [])) | set(atk["intervention"].get("inject", {}))
        sets.append(atk); excluded |= neurons
    return sets


# ====================================================================== DEFENDER
class Defender:
    def __init__(self, behaviour, attack, log, budget=DEFENCE_RULES["budget"], seeds=CHECK_SEEDS):
        self.b, self.attack, self.log, self.budget, self.seeds = behaviour, attack, log, budget, seeds
        self.used = 0
        self.fixed = attack["intervention"]                   # the attack, which cannot be undone

    def _test(self, iv):
        """Two-condition test: on with stimulus, off without. Screened on the check seeds; ACCEPTED only if it also
        holds on every ROBUST_SEED, so minimisation cannot strip a repair to the edge of the ON bar."""
        self.used += 1
        def run(seeds):
            ons, offs = [], []
            for s in seeds:
                _, on, off = self.b.is_normal(seed=s, **merge_iv(self.fixed, iv)); ons.append(on); offs.append(off)
            return ons, offs
        ons, offs = run(self.seeds)
        ok = all(o >= self.b.on_thr for o in ons) and all(o <= self.b.off_thr for o in offs)
        if ok:
            extra = [s for s in ROBUST_SEEDS if s not in self.seeds]
            if extra:
                self.used += 1; e_on, e_off = run(extra)
                ok = all(o >= self.b.on_thr for o in e_on) and all(o <= self.b.off_thr for o in e_off)
        return ok, float(np.mean(ons)), float(np.mean(offs))

    def _try(self, name, hypothesis, iv):
        ok, on, off = self._test(iv)
        self.log.add(name, self.used, hypothesis, _fmt(iv), f"on {on:.1f} / off {off:.1f} Hz", ok)
        return ok, on, off

    def survivors(self):
        """Neurons still firing after the attack that excite the readout, ranked by rate x synapses."""
        _, r = self.b.measure(True, **self.fixed)
        col = readout_col(self.b)
        dead = {int(i) for i in self.fixed.get("lesion", [])}
        score = {i: hz * col[i] for i, hz in r["active"].items() if col[i] > 0 and i not in dead and i not in self.b.protected}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def attacked(self):
        """Neurons the attacker touched: the defender may never modify these (the attack cannot be undone)."""
        f = self.fixed
        # int(): an attack read back from an atlas JSON has string keys ('80293'); normalise so membership tests work
        return {int(i) for i in set(f.get("lesion", [])) | set(f.get("inject", {})) | set(f.get("weaken", {}))}

    def suppressors(self):
        """Inhibitory neurons that fire WITH the stimulus and inhibit the readout (directly or via its drivers),
        not touched by the attacker: candidates for the defender's 'silence' move."""
        _, r = self.b.measure(True, **self.fixed)
        W = self.b.brain.W; ro = self.b.readout
        col = readout_col(self.b); exc = np.flatnonzero(col > 0)
        no = self.attacked() | self.b.protected
        score = {}
        for i, hz in r["active"].items():
            if i in no: continue
            direct = -col[i] if col[i] < 0 else 0.0
            row = W[i, exc].toarray().ravel(); indirect = -row[row < 0].sum() if row.size else 0.0
            v = hz * (direct + 0.5 * indirect)
            if v > 0: score[int(i)] = v
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def inhibitory_inputs(self):
        """Inhibitory synapses onto the readout that are ACTIVE with the stimulus (incl. attacker-injected ones), ranked
        by rate x synapses: candidates for the 'depress' move."""
        _, r = self.b.measure(True, **self.fixed)
        col = readout_col(self.b)
        score = {int(i): hz * -col[i] for i, hz in r["active"].items() if col[i] < 0}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def probe(self, repair, max_size=2, budget=12):
        """Adversarial probe: can the attacker undo attack+repair with <= max_size extra neurons?  Returns the
        attack found (None = the repair survived). Uses a throw-away log so the probe does not clutter the notebook."""
        probe_log = Log(quiet=True)
        A = Attacker(self.b, probe_log, "block", budget=budget, fixed=merge_iv(self.fixed, repair))
        atk = A.greedy(name="probe")
        if atk and atk["size"] <= max_size: return atk
        A = Attacker(self.b, probe_log, "block", budget=budget, fixed=merge_iv(self.fixed, repair))
        atk = A.inject_inhibitors(name="probe")
        return atk if atk and atk["size"] <= max_size else None

    def greedy(self, name="defender-greedy"):
        surv = self.survivors()
        mv = DEFENCE_RULES["moves"]["bias"]["max_mv"]; maxn = DEFENCE_RULES["moves"]["bias"]["max_neurons"]
        maxf = DEFENCE_RULES["moves"]["potentiate"]["max_factor"]; ro = self.b.readout
        plans = [("bias only", surv, lambda S: {"bias": {i: mv for i in S}}),
                 ("potentiate only", surv, lambda S: {"potentiate": onto_readouts(self.b, S, maxf)}),
                 ("bias + potentiate", surv, lambda S: {"bias": {i: mv for i in S}, "potentiate": onto_readouts(self.b, S, min(maxf, 2.0))})]
        # against an INJECT attack the readout can simply stop listening to the hijacked wire: postsynaptic depression
        # on the attacker's own synapses onto the readout. It does not undo the attack -- the neuron still fires at
        # 150 Hz -- and it is the only move that beats a neuron wired straight into the readout.
        W = self.b.brain.W; DEEP = DEFENCE_RULES["moves"]["depress"].get("deep_factor", 0.1)
        hij = [i for i in (int(x) for x in self.attacked()) if any(W[i, r] != 0 for r in self.b.readouts)]
        if hij:
            plans.append(("stop listening to the hijacked wire", hij,
                          lambda S: {"potentiate": {(i, r): DEEP for i in S for r in self.b.readouts if W[i, r] != 0}}))
        best = None
        for label, cands, make in plans:                           # every plan is tried; the smallest repair wins
            for k in (1, 2, 4, 6, 8, maxn):
                if self.used >= self.budget or k > len(cands): break
                if best and k >= best["size"]: break               # cannot beat the current best
                S = cands[:k]; ok, on, off = self._try(name, f"{label}: {k} neurons", make(S))
                if ok:
                    for i in list(S):                              # minimise
                        if self.used >= self.budget: break
                        T = [j for j in S if j != i]
                        if not T: continue
                        ok2, on2, off2 = self._try(name, f"is neuron {i} needed for the repair?", make(T))
                        if ok2: S, on, off = T, on2, off2
                    iv = make(S)
                    cand = {"strategy": name, "move": label, "intervention": iv, "size": size_of(iv),
                            "on_hz": on, "off_hz": off, "simulations": self.used}
                    if best is None or (cand["size"], -cand["on_hz"]) < (best["size"], -best["on_hz"]): best = cand
                    break
        return best

    def robust(self, name="defender-robust", probe_size=2):
        self.budget = max(self.budget, DEFENCE_RULES.get("robust_budget", 90))
        """The stronger defender. Three changes over greedy():
           1. an extra move, SILENCE: lesion up to DEFENCE_RULES['moves']['silence']['max'] inhibitory neurons that are
              suppressing the readout (never neurons the attacker touched);
           2. SPREAD repairs: the same total boost over more neurons (bias at half strength on 2x the neurons), so that
              no single boosted neuron becomes a new choke point;
           3. every candidate repair is probed by a small attacker; a repair the attacker can undo with <= probe_size
              neurons is 'fragile'. The smallest SURVIVING repair wins; if none survives, the smallest repair is
              returned with fragile=True.  Cost = neurons touched (silenced neurons count)."""
        surv = self.survivors(); sup = self.suppressors()
        mv = DEFENCE_RULES["moves"]["bias"]["max_mv"]; maxn = DEFENCE_RULES["moves"]["bias"]["max_neurons"]
        maxf = DEFENCE_RULES["moves"]["potentiate"]["max_factor"]; ro = self.b.readout
        n_sil = DEFENCE_RULES["moves"].get("silence", {}).get("max", 3)
        dep = DEFENCE_RULES["moves"].get("depress", {}).get("min_factor", 0.33); inh = self.inhibitory_inputs()
        ro_mv = DEFENCE_RULES.get("readout_bias_mv", 0.0)
        plans = ([("bias the readout itself", lambda S: {"bias": {r: ro_mv for r in self.b.readouts}}, [ro], (1,)),
                  ("readout bias + depress", lambda S: {"bias": {r: ro_mv for r in self.b.readouts}, "potentiate": onto_readouts(self.b, [i for i in S if i != ro], dep)}, [ro] + inh, (2, 3, 5))]
                 if ro_mv > 0 else []) + \
                [("depress inhibitory synapses onto readout", lambda S: {"potentiate": onto_readouts(self.b, S, dep)}, inh, (1, 2, 3, 4, 6, 8, 12)),
                 ("depress + bias", lambda S: {"potentiate": onto_readouts(self.b, S[:2], dep), "bias": {i: mv for i in S[2:]}}, inh[:2] + surv, (3, 4, 6, 8)),
                 ("silence suppressors", lambda S: {"lesion": list(S)}, sup, (1, 2, n_sil)),
                 ("bias only", lambda S: {"bias": {i: mv for i in S}}, surv, (2, 4, 6, 8, maxn)),
                 ("spread bias (half strength, more neurons)", lambda S: {"bias": {i: mv / 2 for i in S}}, surv, (4, 6, 8, maxn)),
                 ("potentiate only", lambda S: {"potentiate": onto_readouts(self.b, S, maxf)}, surv, (2, 4, 6, 8, maxn)),
                 ("spread potentiate (x1.5, more neurons)", lambda S: {"potentiate": onto_readouts(self.b, S, min(maxf, 1.5))}, surv, (4, 6, 8, maxn)),
                 ("bias + potentiate", lambda S: {"bias": {i: mv for i in S}, "potentiate": onto_readouts(self.b, S, min(maxf, 2.0))}, surv, (2, 4, 6, 8, maxn)),
                 ("silence + bias", lambda S: {"lesion": list(S[:1]), "bias": {i: mv for i in S[1:]}}, sup[:1] + surv, (3, 5, 7))]
        cands = []
        def probe_and_record(label, iv, on, off, note=""):
            undo = self.probe(iv, max_size=probe_size)
            cand = {"strategy": name, "move": label + note, "intervention": iv, "size": size_of(iv), "on_hz": on, "off_hz": off,
                    "simulations": self.used, "fragile": undo is not None, "undone_by": (undo["size"] if undo else None)}
            self.log.add(name, self.used, f"probe: can {undo['size'] if undo else '>' + str(probe_size)} neurons undo '{label}{note}' on {size_of(iv)}?",
                         _fmt(iv), "FRAGILE" if undo else "survives", not undo)
            cands.append(cand); return cand
        for label, make, pool, ks in plans:
            for k in ks:
                if self.used >= self.budget or k > len(pool): break
                S = pool[:k]; ok, on, off = self._try(name, f"{label} on {k}", make(S))
                if not ok: continue
                for i in list(S):                                  # minimise: the cheapest repair (may be fragile)
                    if self.used >= self.budget: break
                    T = [j for j in S if j != i]
                    if not T: continue
                    ok2, on2, off2 = self._try(name, f"is neuron {i} needed?", make(T))
                    if ok2: S, on, off = T, on2, off2
                cand = probe_and_record(label, make(S), on, off)
                if cand["fragile"]:                                # grow with deliberate redundancy until the probe fails
                    for k2 in range(len(S) + 2, min(len(pool), maxn) + 1, 2):
                        if self.used >= self.budget: break
                        S2 = list(dict.fromkeys(S + pool[:k2]))[:k2]
                        ok2, on2, off2 = self._try(name, f"{label}: add redundancy, {k2} neurons", make(S2))
                        if not ok2: continue
                        c2 = probe_and_record(label, make(S2), on2, off2, note=f" + redundancy")
                        if not c2["fragile"]: break
                break                                              # next plan
        if not cands: return None
        robust = [c for c in cands if not c["fragile"]]
        pick = min(robust or cands, key=lambda c: (c["size"], -c["on_hz"]))
        pick["alternatives"] = [(c["move"], c["size"], "fragile" if c["fragile"] else "robust") for c in cands]
        pick["minimal_size"] = min(c["size"] for c in cands)     # the cheapest (fragile) repair, for the redundancy cost
        return pick

    # ---- repair after a CORRUPT attack -------------------------------------------------------------------------
    def suppressors_of(self, stim_on=True):
        """Inhibitory neurons that are active under the attacked stimulus and inhibit the readout, directly or through
        its excitatory drivers. Candidates for 'boost the brake'. Never neurons the attacker touched."""
        _, r = self.b.measure(stim_on, **self.fixed)
        W = self.b.brain.W; ro = self.b.readout
        col = readout_col(self.b); exc = np.flatnonzero(col > 0)
        no = self.attacked() | self.b.protected
        score = {}
        for i, hz in r["active"].items():
            i = int(i)
            if i in no: continue
            direct = -col[i] if col[i] < 0 else 0.0
            row = W[i, exc].toarray().ravel(); indirect = -row[row < 0].sum() if row.size else 0.0
            v = max(hz, 1.0) * (direct + 0.5 * indirect)          # a silent brake is still worth biasing
            if v > 0: score[i] = v
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def drivers_now(self):
        """Excitatory neurons firing under the attacked stimulus that push the readout: candidates for 'weaken'.
        Never neurons the attacker touched (its work cannot be undone) and never the readout itself."""
        _, r = self.b.measure(True, **self.fixed)
        col = readout_col(self.b)
        no = self.attacked() | self.b.protected
        score = {int(i): hz * col[int(i)] for i, hz in r["active"].items() if col[int(i)] > 0 and int(i) not in no}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def _test_off(self, iv, control):
        """The corrupt-repair test, on every seed:
             1. the refused stimulus must NOT drive the readout any more (<= off_thr)  -- the repair worked;
             2. nothing must fire with no stimulus at all (<= off_thr)                 -- no silencing-by-sabotage;
             3. the control behaviour must still pass (>= its on_thr)                  -- the fly must still eat sugar.
        Without 3, the defender could 'repair' bitter-refusal by breaking feeding altogether."""
        self.used += 1
        full = merge_iv(self.fixed, iv)
        def run(seeds):
            bad = [self.b.measure(True, seed=s, **full)[0] for s in seeds]
            quiet = [self.b.measure(False, seed=s, **full)[0] for s in seeds]
            ctrl = [control.measure(True, seed=s, **full)[0] for s in seeds] if control else [np.inf]
            ok = (all(v <= self.b.off_thr for v in bad) and all(v <= self.b.off_thr for v in quiet)
                  and (control is None or all(v >= control.on_thr for v in ctrl)))
            return ok, bad, ctrl
        ok, bad, ctrl = run(self.seeds)
        if ok:                                            # accepted only if it holds on every ROBUST_SEED
            extra = [s for s in ROBUST_SEEDS if s not in self.seeds]
            if extra: self.used += 1; ok = run(extra)[0]
        return ok, float(np.mean(bad)), float(np.mean(ctrl))

    # ---- repair to a TARGET LEVEL, not just past the bar ------------------------------------------------------------
    def healthy_rate(self, seeds=None):
        """Mean readout rate of the unmodified brain with the stimulus (what 100 % recovery means)."""
        return float(np.mean([self.b.measure(True, seed=s)[0] for s in (seeds or ROBUST_SEEDS)]))

    def _level(self, iv, seeds):
        """Readout rates with and without the stimulus, on the attacked brain plus `iv`."""
        full = merge_iv(self.fixed, iv); self.used += 1
        on = [self.b.measure(True, seed=s, **full)[0] for s in seeds]
        off = [self.b.measure(False, seed=s, **full)[0] for s in seeds]
        return on, off

    def restore(self, name="defender-restore", target=0.8, overshoot=1.3, ladder=(1, 2, 3, 4, 6, 8, 12)):
        """Repair to a LEVEL: the behaviour must come back to at least `target` x healthy on EVERY robust seed, and
        never above `overshoot` x healthy (a fly that fires harder than normal is not repaired either).

        greedy() finds the fewest neurons that clear the ON bar; restore() targets a recovery LEVEL instead and asks
        what it costs to bring the behaviour back.

        Moves: bias + potentiate on the strongest surviving drivers, at four strengths, optionally combined with
        stop-listening on the attacker's hijacked wires. Returns the cheapest repair meeting the target, and in
        every case the full COST-RECOVERY CURVE: for each repair size, the best worst-seed recovery reached."""
        H = self.healthy_rate(); lo, hi = target * H, overshoot * H
        surv = self.survivors(); W = self.b.brain.W
        mv = DEFENCE_RULES["moves"]["bias"]["max_mv"]; maxf = DEFENCE_RULES["moves"]["potentiate"]["max_factor"]
        DEEP = DEFENCE_RULES["moves"]["depress"].get("deep_factor", 0.1)
        hij = [i for i in sorted(self.attacked()) if any(W[i, r] != 0 for r in self.b.readouts)]   # attacker wired into the readout
        strengths = [(mv * s, 1.0 + (maxf - 1.0) * s) for s in (0.25, 0.5, 0.75, 1.0)]

        def make(S, bmv, pf, listen):
            iv = {"bias": {i: bmv for i in S}, "potentiate": onto_readouts(self.b, S, pf)}
            if listen and hij: iv["potentiate"].update(onto_readouts(self.b, hij, DEEP))
            return iv

        curve, best, fallback = [], None, None      # fallback: the best repair that WORKS (clears the ON bar on every
        for k in ladder:                             # seed) when none reaches the target level
            if k > len(surv): break
            S = surv[:k]; best_k = None
            for listen in ([False, True] if hij else [False]):
                for bmv, pf in strengths:
                    if self.used >= max(self.budget, 120): break
                    iv = make(S, bmv, pf, listen)
                    on, off = self._level(iv, ROBUST_SEEDS)
                    worst, peak = min(on), max(on)
                    ok = worst >= lo and peak <= hi and all(o <= self.b.off_thr for o in off)
                    self.log.add(name, self.used, f"{k} drivers, bias {bmv:.1f} mV, x{pf:.1f}" + (" + stop listening" if listen else ""),
                                 _fmt(iv), f"worst {worst:.0f} / best {peak:.0f} Hz ({worst / H * 100:.0f}-{peak / H * 100:.0f} %)", ok)
                    if peak <= hi and all(o <= self.b.off_thr for o in off) and (best_k is None or worst > best_k["worst_hz"]):
                        best_k = {"size": size_of(iv), "worst_hz": worst, "mean_hz": float(np.mean(on)), "recovery": worst / H, "intervention": iv}
                    works = worst >= self.b.on_thr and peak <= hi and all(o <= self.b.off_thr for o in off)
                    if works and (fallback is None or worst > fallback["worst_hz"]):
                        fallback = {"strategy": name, "move": f"bias {bmv:.1f} mV + potentiate x{pf:.1f}" + (" + stop listening" if listen else ""),
                                    "intervention": iv, "size": size_of(iv), "on_hz": float(np.mean(on)), "off_hz": float(np.mean(off)),
                                    "worst_hz": worst, "recovery": worst / H, "healthy_hz": H, "below_target": True}
                    if ok:
                        cand = {"strategy": name, "move": f"bias {bmv:.1f} mV + potentiate x{pf:.1f}" + (" + stop listening" if listen else ""),
                                "intervention": iv, "size": size_of(iv), "on_hz": float(np.mean(on)), "off_hz": float(np.mean(off)),
                                "worst_hz": worst, "recovery": worst / H, "healthy_hz": H, "simulations": self.used}
                        if best is None or (cand["size"], -cand["recovery"]) < (best["size"], -best["recovery"]): best = cand
                        break                                        # weakest strength that meets the target at this size
            if best_k: curve.append({k2: v for k2, v in best_k.items() if k2 != "intervention"})
            if best is not None and best["size"] <= k: break        # nothing cheaper can appear at larger k
        if best is None and fallback is not None:
            # the target level is out of reach: hand back the STRONGEST working repair found, labelled with the level
            # it actually reaches
            fallback["simulations"] = self.used
            fallback["why"] = (f"{target * 100:.0f} % of healthy not reachable with up to {ladder[-1]} drivers; "
                               f"best working repair reaches {fallback['recovery'] * 100:.0f} % on every seed")
            best = fallback
        result = best or {"strategy": name, "partial": True, "healthy_hz": H, "simulations": self.used,
                          "why": f"no repair of up to {ladder[-1]} drivers restores the behaviour on every seed"}
        result["curve"] = curve; result["target"] = target
        return result

    def restore_off(self, name="defender-restore", control=None):
        """Repair after a CORRUPT attack: make the fly refuse the bad stimulus again, without losing the good one.

        Moves, in order of how invasive they are:
          boost the brake  -- bias the inhibitory neurons that gate the readout, so they fire more readily;
          strengthen the brake -- potentiate those inhibitory synapses onto the readout (a signed synapse, so
                                  potentiating an inhibitory one deepens the inhibition);
          weaken the push  -- scale down the outgoing synapses of the excitatory neurons now driving the readout.
        The attacker's own neurons are off limits throughout."""
        sup = self.suppressors_of(True); drv = self.drivers_now(); partial = None
        col = readout_col(self.b)
        hij = {int(i) for i in self.attacked()} | {int(i) for i in drv}                  # ids may come back from JSON as strings
        hij = sorted(hij - {int(i) for i in self.b.protected}, key=lambda i: -col[i])    # strongest wires into the readout
        hij = [i for i in hij if col[i] > 0]
        mv = DEFENCE_RULES["moves"]["bias"]["max_mv"]; maxn = DEFENCE_RULES["moves"]["bias"]["max_neurons"]
        maxf = DEFENCE_RULES["moves"]["potentiate"]["max_factor"]; wk = DEFENCE_RULES["moves"]["depress"]["min_factor"]
        DEEP = DEFENCE_RULES["moves"]["depress"].get("deep_factor", 0.1)     # near-silent synapse, still not a lesion
        ro = self.b.readout
        plans = (
            ("boost the brake", sup, lambda S: {"bias": {i: mv for i in S}}),
            ("strengthen the brake", sup, lambda S: {"potentiate": onto_readouts(self.b, S, maxf)}),
            ("boost + strengthen the brake", sup, lambda S: {"bias": {i: mv for i in S},
                                                             "potentiate": onto_readouts(self.b, S, min(maxf, 2.0))}),
            ("weaken the push", drv, lambda S: {"weaken": {i: wk for i in S}}),
            # last resort: stop the READOUT listening to the hijacked wire. This does not undo the attack (the injected
            # neuron still fires); it is postsynaptic depression at the motor neuron, the mirror of 'potentiate'. It is
            # the only move that can beat an injected neuron wired straight into the readout, so attacked neurons'
            # synapses ARE allowed here -- but the control behaviour must still pass, which is what makes it honest.
            ("stop listening to the hijacked wire", hij, lambda S: {"potentiate": onto_readouts(self.b, S, wk)}),
            ("stop listening, hard", hij, lambda S: {"potentiate": onto_readouts(self.b, S, DEEP)}),
            ("stop listening + boost the brake", hij, lambda S: {"potentiate": onto_readouts(self.b, S, DEEP),
                                                                 "bias": {i: mv for i in sup[:len(S)]}}),
        )
        best = None
        for label, cands, make in plans:
            if not cands: continue
            for k in (1, 2, 4, 6, 8, maxn):
                if self.used >= self.budget or k > len(cands): break
                if best and k >= best["size"]: break
                S = cands[:k]
                ok, bad, ctrl = self._test_off(make(S), control)
                self.log.add(name, self.used, f"{label} on {k} neurons", _fmt(make(S)),
                             f"refused stimulus {bad:.1f} Hz, control {ctrl:.1f} Hz", ok)
                if ok:
                    for i in list(S):                                     # minimise
                        if self.used >= self.budget: break
                        T = [j for j in S if j != i]
                        if not T: continue
                        ok2, bad2, ctrl2 = self._test_off(make(T), control)
                        self.log.add(name, self.used, f"is neuron {i} needed?", _fmt(make(T)),
                                     f"refused stimulus {bad2:.1f} Hz, control {ctrl2:.1f} Hz", ok2)
                        if ok2: S, bad, ctrl = T, bad2, ctrl2
                    iv = make(S)
                    cand = {"strategy": name, "move": label, "intervention": iv, "size": size_of(iv),
                            "on_hz": bad, "off_hz": 0.0, "control_hz": ctrl, "simulations": self.used}
                    if best is None or cand["size"] < best["size"]: best = cand
                    break
                # not a full repair: remember how close it got, and what it cost the control behaviour
                if partial is None or bad < partial["on_hz"]:
                    partial = {"strategy": name, "move": label, "intervention": make(S), "size": size_of(make(S)),
                               "on_hz": bad, "control_hz": ctrl, "partial": True}
        if best: return best
        if partial:                       # the blue team LOST this one: say so, with the numbers that show why
            partial["why"] = (f"no repair restores the refusal ({partial['on_hz']:.1f} Hz, needs <= {self.b.off_thr:.0f}) "
                              f"without pushing the control behaviour below its bar ({partial['control_hz']:.1f} Hz)")
        return partial
