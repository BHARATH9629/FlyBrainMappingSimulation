"""Behaviours as pass/fail tests, plus the attack rules (the 'attack surface').
BEHAVIOURS["female"] and BEHAVIOURS["male"] are kept separate: different animals, different datasets, different naming.
Neuron IDs: sugar GRNs and MN9 from Shiu et al. 2024 (example notebook). More behaviours are added as their IDs are
collected from the FlyWire annotations / papers."""
import numpy as np

SUGAR_IDS = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
             720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
             720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
             720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
             720575940611875570]
MN9_ID = 720575940660219265

FEMALE = {   # FlyWire IDs (18-digit root ids)
    "eat": {
        "stimulus": SUGAR_IDS,          # what the world does
        "readout": MN9_ID,              # what we measure
        "on_threshold": 40.0,           # behaviour present if readout >= this (Hz) with stimulus
        "off_threshold": 10.0,          # and readout <= this without stimulus
        "description": "sugar on the tongue makes the feeding motor neuron MN9 fire",
        "source": "Shiu et al. 2024 Nature (sugar GRN and MN9 ids from the example notebook)",
    },
    # ---- new behaviours (milestone 3). Type names from the Schlegel et al. 2024 annotation table. Thresholds are set
    # from calibrate.py on the healthy brain (ON ~ 40 % of the healthy rate, OFF = max(10 Hz, 2x healthy OFF)).
    "refuse_bitter": {
        "stimulus": {"union": [SUGAR_IDS, {"subclass": "bitter", "side": "L"}]},   # the eat sugar GRNs + left bitter GRNs
        "readout": MN9_ID, "expected": "off",
        "on_threshold": 40.0, "off_threshold": 10.0,    # calibrate.py: sugar+bitter 0.0 Hz on 3 seeds vs sugar alone 107 Hz
        "description": "sugar mixed with bitter should NOT make MN9 fire (bitter gates feeding); goal 'corrupt' = make it fire",
        "source": "bitter subtypes LB1a-d, sugar LB3 per the taste connectome papers (Cell 2026 / bioRxiv 2025.08.25)",
        "batch": False,
    },
    "escape": {
        "stimulus": {"types": ["LC4", "LPLC2"], "side": "L"},            # looming detectors of one eye
        "readout": {"types": ["DNp01"], "side": "L"},                    # giant fiber on the stimulated side
        "on_threshold": 80.0, "off_threshold": 10.0,    # calibrate.py: healthy 233 Hz (L), 0 Hz without; both eyes 263 Hz
        "description": "a looming object drives the giant fiber (DNp01), the escape command neuron",
        "source": "LC4 / LPLC2 -> giant fiber: Ache et al. 2019 Curr Biol; von Reyn et al. 2017 Neuron",
    },
    "drink": {
        "stimulus": {"types": ["LB3"], "subclass": "sugar/water", "side": "L", "exclude_ids": SUGAR_IDS},   # left LB3 taste GRNs that are NOT the 20 sugar cells
        "readout": MN9_ID,
        "on_threshold": 40.0, "off_threshold": 10.0,
        "description": "the other labellar taste GRNs (water / sugar class) also drive MN9: drinking",
        "source": "female annotation lumps sugar and water GRNs as 'LB3 sugar/water' (Schlegel 2024); water-specific typing only in the male (LB3a)",
    },
    "retreat": {
        "stimulus": {"types": ["LC16"], "side": None},                   # LC16 in BOTH eyes: an object approaching head-on
        "readout": {"types": ["MDN"]},                                   # moonwalker descending neurons (backward walking), mean of 4
        "on_threshold": 10.0, "off_threshold": 5.0,     # male calibrate: both eyes 13 +/- 1 Hz (weaker than the female's 42), 0 Hz without
        "description": "an approaching object (LC16) drives the moonwalker descending neurons: the fly backs up",
        "source": "LC16 -> backward walking: Wu et al. 2016 eLife; MDN: Bidaye et al. 2014 Science",
    },
    "turn": {
        "stimulus": {"types": ["LC4", "LPLC2"], "side": "L"},            # looming on the LEFT eye
        "readout": {"types": ["DNa02"], "side": "L"},                    # male: the LEFT-labelled DNa02 fires (110 Hz); right-eye looming -> nothing
        "on_threshold": 40.0, "off_threshold": 10.0,    # LATERALITY DIFFERS from the female (R eye -> DNa02_L): likely a soma-side labelling convention; verify in neuroglancer before claiming
        "description": "a threat on one side drives the opposite steering neuron DNa02: the fly turns away from it",
        "source": "DNa02 steering: Rayshubskiy et al. 2020 bioRxiv; Braun et al. 2024",
        "batch": False,
    },
    "groom": {
        "stimulus": {"regex": "JO-(F|E).*", "side": "L"},                # antennal mechanosensory JO-E/F (touch, gravity/wind)
        "readout": {"superclass": "descending", "regex": "DNg.*"},        # PLACEHOLDER: replaced by calibrate.py picks
        "on_threshold": 40.0, "off_threshold": 10.0,
        "description": "antennal touch drives antennal-grooming descending neurons",
        "source": "JO -> aDN grooming: Hampel et al. 2015 eLife (readout chosen data-driven, see HANDOFF)",
        "batch": False,
    },
}

# MALE fly (MaleCNS, brain + nerve cord, Sep 2026). Neurons are named by MaleCNS cell type, not ID.
# Sugar GRNs = labellar types LB3b + LB3c (match Gr64f, "The complete gustatory connectome", Cell 2026);
# bitter = LB1a-d; water = LB3a. MN9 is typed directly.
MALE = {
    "eat": {
        "stimulus": {"types": ["LB3b", "LB3c"], "side": "L"},   # left labellar sugar GRNs, as in Shiu et al. Fig. 2
        "readout": {"types": ["MN9"], "side": "L"},
        "on_threshold": 40.0, "off_threshold": 10.0,
        "description": "sugar on the tongue makes the feeding motor neuron MN9 fire",
    },
    "eat_bitter": {
        "stimulus": {"types": ["LB1a", "LB1b", "LB1c", "LB1d"], "side": "L"},
        "readout": {"types": ["MN9"], "side": "L"},
        "on_threshold": 40.0, "off_threshold": 10.0,
        "description": "bitter on the tongue should NOT make MN9 fire (healthy = fails the ON test; used for corrupt goals)",
        "batch": False,                 # not a block/force target: skip in batch.py / sensitivity.py
    },
    "refuse_bitter": {
        "stimulus": {"types": ["LB3b", "LB3c", "LB1a", "LB1b", "LB1c", "LB1d"], "side": "L"},   # sugar + bitter together
        "readout": {"types": ["MN9"], "side": "L"}, "expected": "off",
        "on_threshold": 40.0, "off_threshold": 25.0,    # calibrate.py: sugar+bitter 20 +/- 3 Hz vs sugar alone 157 Hz
        "description": "sugar mixed with bitter should NOT make MN9 fire (bitter gates feeding); goal 'corrupt' = make it fire",
        "source": "LB3b/c sugar, LB1a-d bitter (Cell 2026 taste connectome)",
        "batch": False,
    },
    "escape": {
        "stimulus": {"types": ["LC4", "LPLC2"], "side": "L"},
        "readout": {"types": ["DNp01"], "side": "L"},
        "on_threshold": 80.0, "off_threshold": 10.0,    # calibrate.py: healthy 216 Hz (both DNp01), 0 Hz without
        "description": "a looming object drives the giant fiber (DNp01), the escape command neuron",
        "source": "LC4 / LPLC2 -> giant fiber: Ache et al. 2019 Curr Biol; von Reyn et al. 2017 Neuron",
    },
    "drink": {
        "stimulus": {"types": ["LB3a"], "side": None},                   # labellar water GRNs (ppk28 match), both sides (one side alone: 27-160 Hz, seed-unstable)
        "readout": {"types": ["MN9"], "side": "L"},
        "on_threshold": 40.0, "off_threshold": 10.0,
        "description": "water on the tongue (water GRNs) makes the feeding motor neuron MN9 fire: drinking",
        "source": "water GRNs ppk28 / LB3a: Cameron et al. 2010 Nature; taste connectome papers",
    },
    "retreat": {
        "stimulus": {"types": ["LC16"], "side": None},                   # LC16 in BOTH eyes: an object approaching head-on
        "readout": {"types": ["MDN"]},                                   # moonwalker descending neurons (backward walking), mean of 4
        "on_threshold": 10.0, "off_threshold": 5.0,     # male calibrate: both eyes 13 +/- 1 Hz (weaker than the female's 42), 0 Hz without
        "description": "an approaching object (LC16) drives the moonwalker descending neurons: the fly backs up",
        "source": "LC16 -> backward walking: Wu et al. 2016 eLife; MDN: Bidaye et al. 2014 Science",
    },
    "turn": {
        "stimulus": {"types": ["LC4", "LPLC2"], "side": "L"},            # looming on the LEFT eye
        "readout": {"types": ["DNa02"], "side": "L"},                    # male: the LEFT-labelled DNa02 fires (110 Hz); right-eye looming -> nothing
        "on_threshold": 40.0, "off_threshold": 10.0,    # LATERALITY DIFFERS from the female (R eye -> DNa02_L): likely a soma-side labelling convention; verify in neuroglancer before claiming
        "description": "a threat on one side drives the opposite steering neuron DNa02: the fly turns away from it",
        "source": "DNa02 steering: Rayshubskiy et al. 2020 bioRxiv; Braun et al. 2024",
        "batch": False,
    },
    "groom": {
        "stimulus": {"regex": "JO-(F|E).*", "side": "L"},
        "readout": {"superclass": "descending_neuron", "regex": "DNg.*"},   # PLACEHOLDER: replaced by calibrate.py picks
        "on_threshold": 40.0, "off_threshold": 10.0,
        "description": "antennal touch drives antennal-grooming descending neurons",
        "source": "JO -> aDN grooming: Hampel et al. 2015 eLife (readout chosen data-driven, see HANDOFF)",
        "batch": False,
    },
    "courtship": {
        "stimulus": {"receptor": "ppk2", "side": None},                    # pheromone-sensing leg GRNs (ppk23 + ppk25), both sides
        "readout": {"regex": "pC1.*|aSP10.*|P1.*"},                        # male courtship command clusters
        "on_threshold": 40.0, "off_threshold": 10.0,
        "description": "pheromone taste (ppk23 GRNs) drives the male courtship command clusters pC1 / aSP10",
        "source": "ppk23 -> P1/pC1: Thistle et al. 2012 Cell; Clowney et al. 2015 Neuron; types from MaleCNS (male only)",
        "batch": False,
    },
}

BEHAVIOURS = {"female": FEMALE, "male": MALE}

# ---------------------------------------------------------------- attack rules
RULES = {
    "forbidden_targets": "motor_and_readout",   # attacker may not touch the readout neuron or motor neurons
    "moves": {
        "lesion":     {"max": 10},              # silence up to 10 neurons
        "weaken":     {"min_factor": 0.5},      # outgoing synapses may be scaled down to 50 %
        "inject":     {"max_hz": 150.0, "max_neurons": 20},   # false sensory input at realistic rates
    },
    "budget": 40,                                # simulations per agent run
}
DEFENCE_RULES = {
    "moves": {
        "bias":       {"max_mv": 6.0, "max_neurons": 12},     # hold neurons closer to threshold (below it)
        "potentiate": {"max_factor": 3.0, "max_synapses": 12},  # strengthen surviving synapses onto the readout
        "silence":    {"max": 3},                               # lesion inhibitory neurons that suppress the readout (never attacked ones)
        "depress":    {"min_factor": 0.33, "max_synapses": 12}, # weaken inhibitory synapses onto the readout (plasticity, mirror of potentiate)
    },
    "budget": 40,
    "robust_budget": 90,                         # Defender.robust() also probes each candidate with a small attacker
}


class Behaviour:
    """A behaviour = stimulus set, readout set, ON/OFF thresholds.
    Spec fields: stimulus, readout (IDs, or a dict for by_type: types/regex/subclass/superclass/side/exclude_types),
    on_threshold, off_threshold, description, and optionally
      expected  : "on" (default: stimulus should drive the readout) or "off" (stimulus should NOT; e.g. bitter -> MN9),
      side      : override the stimulus side ("L", "R", None = both) -- used by the side controls,
      sim       : {"depth": 3, "min_syn": 5} sub-network settings passed to Brain.run,
      batch     : False to skip in batch.py / sensitivity.py, source: citation string."""
    def __init__(self, brain, name, side="default"):
        cfg = dict(BEHAVIOURS[brain.sex][name]); self.cfg = cfg
        self.name, self.brain = name, brain
        if side != "default":                                  # side control: move stimulus AND a sided readout
            for key in ("stimulus", "readout"):
                if isinstance(cfg[key], dict) and "union" not in cfg[key] and "side" in cfg[key]:
                    cfg[key] = dict(cfg[key], side=side)
        self.stim = self._resolve(cfg["stimulus"])
        self.readouts = [int(i) for i in self._resolve(cfg["readout"])]
        if not self.readouts: raise ValueError(f"{name}: readout resolves to no neurons")
        self.readout = self.readouts[0]                       # single-neuron code paths use this
        self.on_thr, self.off_thr = cfg["on_threshold"], cfg["off_threshold"]
        self.expected = cfg.get("expected", "on")
        self.sim = cfg.get("sim", {})
        self.protected = set(self.readouts)                   # the attacker may not touch these

    def _resolve(self, spec):
        """A list of dataset IDs, a single ID, a by_type query dict, or {"union": [spec, spec, ...]}."""
        if isinstance(spec, dict) and "union" in spec:
            return np.unique(np.concatenate([self._resolve(x) for x in spec["union"]])).astype(int)
        if isinstance(spec, dict) and "exclude_ids" in spec:
            spec = dict(spec); excl = set(self.brain.idx(spec.pop("exclude_ids")).tolist())
            return np.array([i for i in self._resolve(spec) if i not in excl], int)
        if isinstance(spec, dict):
            return self.brain.by_type(types=spec.get("types"), side=spec.get("side"), subclass=spec.get("subclass"),
                                      superclass=spec.get("superclass"), exclude_types=spec.get("exclude_types", ()),
                                      regex=spec.get("regex"), receptor=spec.get("receptor"))
        return self.brain.idx(spec if isinstance(spec, (list, tuple)) else [spec])

    def measure(self, with_stimulus=True, seed=1, **interventions):
        """Mean rate over the readout set (Hz), plus the raw run."""
        kw = dict(self.sim); kw.update(interventions)
        r = self.brain.run(self.stim if with_stimulus else [], record=self.readouts, seed=seed, **kw)
        return float(np.mean([r["rates"][i] for i in self.readouts])), r

    def is_normal(self, **interventions):
        """expected 'on': present with stimulus AND absent without.  expected 'off': absent in both conditions."""
        on, _ = self.measure(True, **interventions); off, _ = self.measure(False, **interventions)
        if self.expected == "off": return on <= self.off_thr and off <= self.off_thr, on, off
        return on >= self.on_thr and off <= self.off_thr, on, off
