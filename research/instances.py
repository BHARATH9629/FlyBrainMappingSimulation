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
    "r_turn_L":   (dict(stimulus={"types": ["LC4", "LPLC2"], "side": LEFT}, readout={"types": ["DNa02"], "side": RIGHT},
                        on_threshold=20.0, off_threshold=10.0, description="left-eye looming -> DNa02 R (turn right, away)"), "confirm"),
}


def register(sex="female"):
    for name, (spec, _) in INSTANCES.items(): BEHAVIOURS[sex][name] = dict(spec)
    return list(INSTANCES)


def role(name): return INSTANCES[name][1]
