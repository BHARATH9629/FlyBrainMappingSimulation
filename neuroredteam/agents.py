"""
Red team (attacker) and blue team (defender), algorithmic versions.  v2: robust.

Shared loop: choose an experiment -> run the brain -> check the behaviour -> goal met on ALL check seeds? -> record,
then try to shrink the intervention set.  An LLM agent plugs into the same tools later; these are the baselines.

Goals
  block  : behaviour must disappear with the stimulus present         (readout <= off_thr)
  force  : behaviour must appear WITHOUT the stimulus                  (readout >= on_thr)
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


def size_of(iv):
    """Cost of an intervention = number of distinct neurons touched."""
    s = set(iv.get("lesion", [])) | set(iv.get("weaken", {})) | set(iv.get("inject", {})) | set(iv.get("bias", {}))
    s |= {a for (a, b) in iv.get("potentiate", {})}
    return len(s)


# ====================================================================== ATTACKER
class Attacker:
    def __init__(self, behaviour, log, goal="block", budget=RULES["budget"], seeds=CHECK_SEEDS, excluded=()):
        self.b, self.log, self.goal, self.budget, self.seeds = behaviour, log, goal, budget, seeds
        self.used = 0; self.excluded = set(excluded)      # neurons a previous attack used (for redundancy search)

    # ---- the referee -------------------------------------------------------
    def _check(self, iv):
        """Goal met on every seed?  Returns (ok, mean readout Hz)."""
        self.used += 1
        with_stim = self.goal == "block"
        rates = [self.b.measure(with_stim, seed=s, **iv)[0] for s in self.seeds]
        ok = all(r <= self.b.off_thr for r in rates) if self.goal == "block" else all(r >= self.b.on_thr for r in rates)
        return ok, float(np.mean(rates))

    def _try(self, name, hypothesis, iv):
        ok, hz = self._check(iv)
        self.log.add(name, self.used, hypothesis, _fmt(iv), f"readout {hz:.1f} Hz", ok)
        return ok, hz

    # ---- candidate generators ---------------------------------------------
    def drivers(self):
        """Excitatory inputs to the readout that are active with the stimulus, ranked by rate x synapses."""
        _, r = self.b.measure(True)
        col = self.b.brain.W[:, self.b.readout].toarray().ravel()
        score = {i: hz * col[i] for i, hz in r["active"].items()
                 if col[i] > 0 and i not in self.b.protected and i not in self.excluded}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def inhibitors(self):
        """Neurons with inhibitory synapses onto the readout, strongest first (they are usually silent: inject wakes them)."""
        col = self.b.brain.W[:, self.b.readout].toarray().ravel()
        return [int(i) for i in np.argsort(col) if col[i] < 0 and i not in self.b.protected and i not in self.excluded][:20]

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
        _, r = self.b.measure(True)
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
            ok, on, off = self.b.is_normal(seed=s, **self.fixed, **iv); ons.append(on); offs.append(off)
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
        score = {i: hz * col[i] for i, hz in r["active"].items() if col[i] > 0 and i not in dead and i != self.b.readout}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def greedy(self, name="defender-greedy"):
        surv = self.survivors()
        mv = DEFENCE_RULES["moves"]["bias"]["max_mv"]; maxn = DEFENCE_RULES["moves"]["bias"]["max_neurons"]
        maxf = DEFENCE_RULES["moves"]["potentiate"]["max_factor"]; ro = self.b.readout
        plans = (("bias only", lambda S: {"bias": {i: mv for i in S}}),
                 ("potentiate only", lambda S: {"potentiate": {(i, ro): maxf for i in S}}),
                 ("bias + potentiate", lambda S: {"bias": {i: mv for i in S}, "potentiate": {(i, ro): min(maxf, 2.0) for i in S}}))
        for label, make in plans:
            for k in (2, 4, 6, 8, maxn):
                if self.used >= self.budget or k > len(surv): break
                S = surv[:k]; ok, on, off = self._try(name, f"{label} on the {k} strongest surviving drivers", make(S))
                if ok:
                    for i in list(S):                              # minimise
                        if self.used >= self.budget: break
                        T = [j for j in S if j != i]
                        if not T: continue
                        ok2, on2, off2 = self._try(name, f"is neuron {i} needed for the repair?", make(T))
                        if ok2: S, on, off = T, on2, off2
                    iv = make(S)
                    return {"strategy": name, "move": label, "intervention": iv, "size": size_of(iv),
                            "on_hz": on, "off_hz": off, "simulations": self.used}
        return None
