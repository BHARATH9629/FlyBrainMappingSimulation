"""Phase 4 behaviour instances, registered IN MEMORY only (the demo's behaviours.py is never edited).

Each instance is a behaviour spec in the same format as neuroredteam/behaviours.py. Importing this module adds them to
behaviours.BEHAVIOURS["female"] under names starting with "r_", for this Python process only.
See research/PREREGISTRATION.md, section 7, for why the taste and bitter instances are symmetric pairs.
"""
import guard  # noqa: F401  (puts neuroredteam/ on the path)
from behaviours import BEHAVIOURS

MN9_R, MN9_L = 720575940660219265, 720575940618238523          # CB0701, the feeding motor neuron on each side
LEFT, RIGHT = "L", "R"


def _taste(side):      return {"types": ["LB3"], "side": side}               # all labellar sugar/water neurons of a side
def _bitter(side):     return {"subclass": "bitter", "side": side}


INSTANCES = {
    # name: (spec, role) -- role: "confirm" counts for H1-H4; "reference" only for the left/right comparison (H5)
    "r_taste_L":  (dict(stimulus=_taste(LEFT), readout=MN9_R, on_threshold=40.0, off_threshold=10.0,
                        description="all left taste (LB3) neurons -> MN9 R"), "reference"),
    "r_taste_R":  (dict(stimulus=_taste(RIGHT), readout=MN9_L, on_threshold=40.0, off_threshold=10.0,
                        description="all right taste (LB3) neurons -> MN9 L"), "confirm"),
    "r_bitter_L": (dict(stimulus={"union": [_taste(LEFT), _bitter(LEFT)]}, readout=MN9_R, expected="off",
                        on_threshold=40.0, off_threshold=10.0, control="r_taste_L",
                        description="left taste + left bitter neurons -> MN9 R must stay quiet"), "reference"),
    "r_bitter_R": (dict(stimulus={"union": [_taste(RIGHT), _bitter(RIGHT)]}, readout=MN9_L, expected="off",
                        on_threshold=40.0, off_threshold=10.0, control="r_feed_L_to_MN9L",
                        description="right taste + right bitter neurons -> MN9 L must stay quiet"), "confirm"),
    # the control for r_bitter_R: normal feeding through the SAME output neuron (MN9 L) must survive any repair
    "r_feed_L_to_MN9L": (dict(stimulus=_taste(LEFT), readout=MN9_L, on_threshold=40.0, off_threshold=10.0,
                              description="all left taste neurons -> MN9 L (control for r_bitter_R)"), "control"),
    "r_escape_R": (dict(stimulus={"types": ["LC4", "LPLC2"], "side": RIGHT}, readout={"types": ["DNp01"], "side": RIGHT},
                        on_threshold=80.0, off_threshold=10.0, description="right-eye looming -> giant fibre DNp01 R"), "confirm"),
    # antennal grooming (Shiu et al. 2024, Fig. 5): Johnston's-organ touch neurons (their JO-CE, JO-F and JO-m lists,
    # left side) -> aDN1 (FlyWire DNg62). groom_R uses the same cell types on the right side and the other aDN1.
    "r_groom_L":  (dict(stimulus="JON_LEFT", readout=720575940616185531, on_threshold=40.0, off_threshold=10.0,
                        description="left antennal touch neurons (Shiu JON list) -> aDN1 (DNg62)"), "confirm"),
    "r_groom_R":  (dict(stimulus="JON_RIGHT_TYPEMATCHED", readout={"types": ["DNg62"], "side": "L"}, on_threshold=40.0,
                        off_threshold=10.0, description="right antennal touch neurons, same types -> the other aDN1"), "confirm"),
    "r_turn_L":   (dict(stimulus={"types": ["LC4", "LPLC2"], "side": LEFT}, readout={"types": ["DNa02"], "side": RIGHT},
                        on_threshold=20.0, off_threshold=10.0, description="left-eye looming -> DNa02 R (turn right, away)"), "confirm"),
}


def _jon_lists():
    import json, os
    d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "grooming_ids_from_shiu_figures.json")))
    return d["neu_JON_CE"] + d["neu_JON_F"] + d["neu_JON_D_m"]


def register(sex="female", brain=None):
    """add the instances to BEHAVIOURS (this process only). The grooming stimulus is resolved to neuron IDs here:
    the left set is the published list; the right set is every right-side neuron of the same cell types."""
    for name, (spec, _) in INSTANCES.items():
        spec = dict(spec)
        if spec["stimulus"] in ("JON_LEFT", "JON_RIGHT_TYPEMATCHED"):
            left = _jon_lists()
            if spec["stimulus"] == "JON_LEFT": spec["stimulus"] = left
            else:
                if brain is None: continue                          # needs the brain to look up types
                idx = brain.idx(left); types = sorted({brain.type_of(i) for i in idx if brain.type_of(i)})
                right = brain.by_type(types=types, side="R")
                spec["stimulus"] = [int(brain.ids[i]) for i in right]
        BEHAVIOURS[sex][name] = spec
    return [n for n in INSTANCES if n in BEHAVIOURS[sex]]


def role(name): return INSTANCES[name][1]
