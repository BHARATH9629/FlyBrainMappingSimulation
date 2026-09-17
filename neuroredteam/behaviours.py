"""Behaviours as pass/fail tests, plus the attack rules (the 'attack surface').
Neuron IDs: sugar GRNs and MN9 from Shiu et al. 2024 (example notebook). More behaviours are added as their IDs are
collected from the FlyWire annotations / papers."""

SUGAR_IDS = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
             720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
             720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
             720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
             720575940611875570]
MN9_ID = 720575940660219265

BEHAVIOURS = {
    "eat": {
        "stimulus": SUGAR_IDS,          # what the world does
        "readout": MN9_ID,              # what we measure
        "on_threshold": 40.0,           # behaviour present if readout >= this (Hz) with stimulus
        "off_threshold": 10.0,          # and readout <= this without stimulus
        "description": "sugar on the tongue makes the feeding motor neuron MN9 fire",
    },
}

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
    },
    "budget": 40,
}


class Behaviour:
    def __init__(self, brain, name):
        cfg = BEHAVIOURS[name]
        self.name, self.brain = name, brain
        self.stim = brain.idx(cfg["stimulus"]); self.readout = int(brain.idx([cfg["readout"]])[0])
        self.on_thr, self.off_thr = cfg["on_threshold"], cfg["off_threshold"]
        self.protected = {self.readout}     # the attacker may not touch these

    def measure(self, with_stimulus=True, seed=1, **interventions):
        r = self.brain.run(self.stim if with_stimulus else [], record=[self.readout], seed=seed, **interventions)
        return r["rates"][self.readout], r

    def is_normal(self, **interventions):
        """Behaviour present with stimulus AND absent without it."""
        on, _ = self.measure(True, **interventions); off, _ = self.measure(False, **interventions)
        return on >= self.on_thr and off <= self.off_thr, on, off
