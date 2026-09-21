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

CHECK_SEEDS = (1, 2, 3)          # every goal is verified on these seeds


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


# ====================================================================== ATTACKER
class Attacker:
    def __init__(self, behaviour, log, goal="block", budget=RULES["budget"], seeds=CHECK_SEEDS, excluded=(), fixed=None):
        self.b, self.log, self.goal, self.budget, self.seeds = behaviour, log, goal, budget, seeds
        self.used = 0; self.excluded = set(excluded)      # neurons a previous attack used (for redundancy search)
        self.fixed = fixed or {}                          # interventions already on the brain (earlier attacks + repairs)
        self.excluded |= set(self.fixed.get("lesion", [])) | set(self.fixed.get("inject", {}))   # do not re-use them

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
        self.log.add(name, self.used, hypothesis, _fmt(iv), f"readout {hz:.1f} Hz", ok)
        return ok, hz

    # ---- candidate generators ---------------------------------------------
    def drivers(self):
        """Excitatory inputs to the readout that are active with the stimulus, ranked by rate x synapses.
        If every direct driver is excluded (threat model T2), fall back one hop upstream: active neurons that excite
        the direct drivers, ranked by rate x (synapses onto drivers, weighted by the drivers' synapses onto the readout)."""
        _, r = self.b.measure(True, **self.fixed)
        W = self.b.brain.W; col = W[:, self.b.readout].toarray().ravel()
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
        col = W[:, ro].toarray().ravel()
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
        W = self.b.brain.W; col = W[:, self.b.readout].toarray().ravel()
        out = [int(i) for i in np.argsort(col) if col[i] < 0 and i not in self.b.protected and i not in self.excluded][:20]
        if not out and self.excluded:
            inh = [int(i) for i in np.argsort(col) if col[i] < 0][:20]
            sub = W[:, inh].toarray(); wi = -col[inh]
            v = (np.clip(sub, 0, None) * wi).sum(axis=1)
            out = [int(i) for i in np.argsort(-v) if v[i] > 0 and i not in self.b.protected and i not in self.excluded][:20]
        return out

    def excitatory_inputs(self):
        col = self.b.brain.W[:, self.b.readout].toarray().ravel()
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
        for k in (1, 2, 3, 5, 8, 12):
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
        """Two-condition test on every seed: on with stimulus, off without."""
        self.used += 1
        ons, offs = [], []
        for s in self.seeds:
            ok, on, off = self.b.is_normal(seed=s, **merge_iv(self.fixed, iv)); ons.append(on); offs.append(off)
        ok = all(o >= self.b.on_thr for o in ons) and all(o <= self.b.off_thr for o in offs)
        return ok, float(np.mean(ons)), float(np.mean(offs))

    def _try(self, name, hypothesis, iv):
        ok, on, off = self._test(iv)
        self.log.add(name, self.used, hypothesis, _fmt(iv), f"on {on:.1f} / off {off:.1f} Hz", ok)
        return ok, on, off

    def survivors(self):
        """Neurons still firing after the attack that excite the readout, ranked by rate x synapses."""
        _, r = self.b.measure(True, **self.fixed)
        col = self.b.brain.W[:, self.b.readout].toarray().ravel()
        dead = set(self.fixed.get("lesion", []))
        score = {i: hz * col[i] for i, hz in r["active"].items() if col[i] > 0 and i not in dead and i not in self.b.protected}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def attacked(self):
        """Neurons the attacker touched: the defender may never modify these (the attack cannot be undone)."""
        f = self.fixed
        return set(f.get("lesion", [])) | set(f.get("inject", {})) | set(f.get("weaken", {}))

    def suppressors(self):
        """Inhibitory neurons that fire WITH the stimulus and inhibit the readout (directly or via its drivers),
        not touched by the attacker: candidates for the defender's 'silence' move."""
        _, r = self.b.measure(True, **self.fixed)
        W = self.b.brain.W; ro = self.b.readout
        col = W[:, ro].toarray().ravel(); exc = np.flatnonzero(col > 0)
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
        col = self.b.brain.W[:, self.b.readout].toarray().ravel()
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
        plans = (("bias only", lambda S: {"bias": {i: mv for i in S}}),
                 ("potentiate only", lambda S: {"potentiate": {(i, ro): maxf for i in S}}),
                 ("bias + potentiate", lambda S: {"bias": {i: mv for i in S}, "potentiate": {(i, ro): min(maxf, 2.0) for i in S}}))
        best = None
        for label, make in plans:                                  # every plan is tried; the smallest repair wins
            for k in (2, 4, 6, 8, maxn):
                if self.used >= self.budget or k > len(surv): break
                if best and k >= best["size"]: break               # cannot beat the current best
                S = surv[:k]; ok, on, off = self._try(name, f"{label} on the {k} strongest surviving drivers", make(S))
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
        plans = ([("bias the readout itself", lambda S: {"bias": {ro: ro_mv}}, [ro], (1,)),
                  ("readout bias + depress", lambda S: {"bias": {ro: ro_mv}, "potentiate": {(i, ro): dep for i in S if i != ro}}, [ro] + inh, (2, 3, 5))]
                 if ro_mv > 0 else []) + \
                [("depress inhibitory synapses onto readout", lambda S: {"potentiate": {(i, ro): dep for i in S}}, inh, (1, 2, 3, 4, 6, 8, 12)),
                 ("depress + bias", lambda S: {"potentiate": {(i, ro): dep for i in S[:2]}, "bias": {i: mv for i in S[2:]}}, inh[:2] + surv, (3, 4, 6, 8)),
                 ("silence suppressors", lambda S: {"lesion": list(S)}, sup, (1, 2, n_sil)),
                 ("bias only", lambda S: {"bias": {i: mv for i in S}}, surv, (2, 4, 6, 8, maxn)),
                 ("spread bias (half strength, more neurons)", lambda S: {"bias": {i: mv / 2 for i in S}}, surv, (4, 6, 8, maxn)),
                 ("potentiate only", lambda S: {"potentiate": {(i, ro): maxf for i in S}}, surv, (2, 4, 6, 8, maxn)),
                 ("spread potentiate (x1.5, more neurons)", lambda S: {"potentiate": {(i, ro): min(maxf, 1.5) for i in S}}, surv, (4, 6, 8, maxn)),
                 ("bias + potentiate", lambda S: {"bias": {i: mv for i in S}, "potentiate": {(i, ro): min(maxf, 2.0) for i in S}}, surv, (2, 4, 6, 8, maxn)),
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
