"""
Red team (attacker) and blue team (defender) agents, algorithmic versions.
Both share the same loop:  choose an experiment -> run the brain -> check the behaviour -> goal met? -> record.
An LLM version plugs into the same tools later; these are the baselines it must beat.
"""
import numpy as np, time
from behaviours import RULES, DEFENCE_RULES


class Log:
    def __init__(self): self.rows = []; self.t0 = time.time()
    def add(self, agent, step, hypothesis, intervention, result, ok):
        self.rows.append(dict(agent=agent, step=step, hypothesis=hypothesis, intervention=intervention,
                              result=result, goal_met=ok, t=round(time.time() - self.t0, 1)))
        tag = "GOAL" if ok else "    "
        print(f"  [{agent} {step:2d}] {tag} {hypothesis:<58s} {intervention:<38s} -> {result}")


# ====================================================================== ATTACKER
class Attacker:
    """Goal 'block': behaviour must disappear with the stimulus present (readout <= off_threshold).
       Allowed moves: lesion / weaken interneurons (never the readout), inject false input."""
    def __init__(self, behaviour, log, budget=RULES["budget"]):
        self.b, self.log, self.budget, self.used = behaviour, log, budget, 0

    def _blocked(self, **iv):
        self.used += 1
        on, r = self.b.measure(True, **iv)
        return on <= self.b.off_thr, on, r

    def candidates(self):
        """Rank interneurons by how much they drive the readout right now: rate x excitatory synapses onto it."""
        on, r = self.b.measure(True)
        col = self.b.brain.W[:, self.b.readout].toarray().ravel()
        score = {i: hz * col[i] for i, hz in r["active"].items() if col[i] > 0 and i not in self.b.protected}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def greedy(self, name="attacker-greedy"):
        """Cut the strongest driver, then the next, ... until the behaviour is blocked. Then try to shrink the set."""
        cands = self.candidates(); chosen = []
        for i in cands[:RULES["moves"]["lesion"]["max"]]:
            if self.used >= self.budget: break
            chosen.append(i)
            ok, on, _ = self._blocked(lesion=chosen)
            self.log.add(name, self.used, f"cut the {len(chosen)} strongest surviving drivers of the readout",
                         f"lesion {len(chosen)} neurons", f"readout {on:.1f} Hz", ok)
            if ok: break
        else:
            return None
        # minimise: try dropping each neuron from the set
        for i in list(chosen):
            if self.used >= self.budget: break
            trial = [j for j in chosen if j != i]
            if not trial: continue
            ok, on, _ = self._blocked(lesion=trial)
            self.log.add(name, self.used, f"is neuron {i} necessary? try without it", f"lesion {len(trial)} neurons",
                         f"readout {on:.1f} Hz", ok)
            if ok: chosen = trial
        return {"move": "lesion", "neurons": [int(i) for i in chosen], "size": len(chosen), "simulations": self.used}

    def random(self, name="attacker-random", rng=np.random.default_rng(0)):
        """Same rules, no intelligence: cut random active interneurons."""
        on, r = self.b.measure(True)
        pool = [i for i in r["active"] if i not in self.b.protected]
        best = None
        while self.used < self.budget:
            k = rng.integers(1, RULES["moves"]["lesion"]["max"] + 1)
            trial = [int(i) for i in rng.choice(pool, k, replace=False)]
            ok, on, _ = self._blocked(lesion=trial)
            self.log.add(name, self.used, f"random guess of {k} active neurons", f"lesion {k} neurons", f"readout {on:.1f} Hz", ok)
            if ok and (best is None or k < best["size"]):
                best = {"move": "lesion", "neurons": trial, "size": k, "simulations": self.used}
        return best


# ====================================================================== DEFENDER
class Defender:
    """Given the attacked brain (lesion fixed, cannot be undone), restore the behaviour with the fewest
       stimulated neurons. Must pass BOTH tests: on with stimulus, off without (no forcing).
       Allowed moves: bias (hold neurons a few mV closer to threshold), potentiate synapses onto the readout."""
    def __init__(self, behaviour, attack, log, budget=DEFENCE_RULES["budget"]):
        self.b, self.attack, self.log, self.budget, self.used = behaviour, attack, log, budget, 0
        self.lesion = attack["neurons"]

    def _test(self, **iv):
        self.used += 1
        ok, on, off = self.b.is_normal(lesion=self.lesion, **iv)
        return ok, on, off

    def survivors(self):
        """Neurons that still fire after the attack AND excite the readout, ranked by rate x synapses."""
        on, r = self.b.measure(True, lesion=self.lesion)
        col = self.b.brain.W[:, self.b.readout].toarray().ravel()
        score = {i: hz * col[i] for i, hz in r["active"].items() if col[i] > 0 and i not in self.lesion}
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])]

    def greedy(self, name="defender-greedy"):
        surv = self.survivors()
        maxn = DEFENCE_RULES["moves"]["bias"]["max_neurons"]; mv = DEFENCE_RULES["moves"]["bias"]["max_mv"]
        maxf = DEFENCE_RULES["moves"]["potentiate"]["max_factor"]
        # escalate: bias only -> potentiate only -> both, growing the set each time
        for label, make in (("bias only", lambda S: dict(bias={i: mv for i in S})),
                            ("potentiate only", lambda S: dict(potentiate={(i, self.b.readout): maxf for i in S})),
                            ("bias + potentiate", lambda S: dict(bias={i: mv for i in S},
                                                                 potentiate={(i, self.b.readout): min(maxf, 2.0) for i in S}))):
            for k in (2, 4, 6, 8, maxn):
                if self.used >= self.budget: return None
                S = surv[:k]; iv = make(S)
                ok, on, off = self._test(**iv)
                self.log.add(name, self.used, f"{label} on the {k} strongest surviving drivers", f"{label}, {k} neurons",
                             f"on {on:.1f} Hz / off {off:.1f} Hz", ok)
                if ok:
                    # minimise: drop neurons one at a time
                    for i in list(S):
                        if self.used >= self.budget: break
                        T = [j for j in S if j != i]
                        if not T: continue
                        ok2, on2, off2 = self._test(**make(T))
                        self.log.add(name, self.used, f"is neuron {i} needed for the repair?", f"{label}, {len(T)} neurons",
                                     f"on {on2:.1f} Hz / off {off2:.1f} Hz", ok2)
                        if ok2: S = T
                    return {"move": label, "neurons": [int(i) for i in S], "size": len(S), "on_hz": on, "off_hz": off,
                            "simulations": self.used}
        return None
