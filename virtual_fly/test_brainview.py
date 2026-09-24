"""Is every firing neuron actually visible in the brain panel?

Runs a real behaviour with spikes, renders the brain panel after the whole run, and checks each neuron that fired:
the pixel at its position must be clearly brighter than the empty background at that pixel. Neurons under the text
overlays (title / counts at the top, legend at the bottom) are counted separately. Neurons with no recorded
position cannot be drawn at all and must be reported by BrainView.missing.
Optionally compares against an older brainview.py (--old path) to show the difference.
Usage: python virtual_fly/test_brainview.py [--old C:/path/to/old_brainview.py] [--save out/brainview_check]
"""
import argparse, importlib.util, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "neuroredteam")); sys.path.insert(0, HERE)
from brain import Brain
from behaviours import Behaviour
import brainview as new_bv

ap = argparse.ArgumentParser()
ap.add_argument("--old", default="", help="path to an older brainview.py to compare with")
ap.add_argument("--save", default=os.path.join(HERE, "out", "brainview_check"))
args = ap.parse_args()
os.makedirs(args.save, exist_ok=True)
MIN_GAIN = 40          # a visible dot must add at least this much brightness (sum of R+G+B, 0-765) over the background

brain = Brain("female")
views = {"new": new_bv.BrainView(brain, 640, 480)}
if args.old:
    spec = importlib.util.spec_from_file_location("old_bv", args.old); old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old); views["old"] = old.BrainView(brain, 640, 480)

ok_all = True
for name in ("eat", "escape"):
    b = Behaviour(brain, name)
    r = brain.run(b.stim, record=b.readouts, seed=1, spikes=True, **dict(b.sim))
    fired = np.unique(r["spike_i"])
    for tag, v in views.items():
        v.reset(); v.advance(300.0, r["spike_t"], r["spike_i"])
        kw = dict(stimulus=b.stim, readout=b.readout, title="t", subtitle="s")
        if tag == "new": kw["counts"] = f"{fired.size:,} neurons fired"
        img = v.frame(**kw).astype(int)
        base = (np.clip(v.base, 0, 1) * 255).astype(int)
        drawable = fired[v.ok[fired]]
        px, py = v.px[drawable], v.py[drawable]
        gain = img[py, px].sum(1) - base[py, px].sum(1)
        under_text = (py < 70) | (py > v.H - 50)
        visible = gain >= MIN_GAIN
        n_bad = int((~visible & ~under_text).sum())
        missing = int((~v.ok[fired]).sum())
        print(f"{name:7s} [{tag}]  fired {fired.size:,} | no position {missing} | drawn {drawable.size:,} | "
              f"visible {int(visible.sum()):,} ({100 * visible.mean():.1f} %) | invisible outside text {n_bad} | "
              f"under text overlays {int(under_text.sum())}", flush=True)
        if tag == "new":
            if n_bad: ok_all = False
            if getattr(v, "missing", None) != int((~v.ok).sum()): ok_all = False; print("   ! BrainView.missing is wrong")
        from PIL import Image
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(args.save, f"{name}_{tag}.png"))
print("PASS: every drawable firing neuron is visible" if ok_all else "FAIL: some firing neurons are not visible")
