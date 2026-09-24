"""Checks for the rotatable 3-D brain view.

  1. FRONT VIEW UNCHANGED: at zero rotation / zoom 1 every neuron lands on the same pixel as in the previous
     (fixed-view) BrainView.
  2. FULL TURN: rotating 360 degrees returns every neuron to its start pixel.
  3. VISIBLE AFTER ROTATING: at several views, every firing neuron that is on screen is clearly brighter than the
     background (same rule as test_brainview.py); neurons pushed off the panel by zooming are counted.
  4. SPEED: time to re-project the brain for a new view, and to render a frame.
Saves PNGs of the rotated views to out/rotation_check/ for a visual look.
Usage: python virtual_fly/test_rotation.py --old C:/path/to/brainview_before_step3.py
"""
import argparse, importlib.util, os, sys, time
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "neuroredteam")); sys.path.insert(0, HERE)
from brain import Brain
from behaviours import Behaviour
from brainview import BrainView

ap = argparse.ArgumentParser(); ap.add_argument("--old", required=True)
args = ap.parse_args()
OUT = os.path.join(HERE, "out", "rotation_check"); os.makedirs(OUT, exist_ok=True)
MIN_GAIN = 40
brain = Brain("female"); ok_all = True

# 1. front view unchanged
spec = importlib.util.spec_from_file_location("old_bv", args.old); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
vo, vn = old.BrainView(brain, 640, 480), BrainView(brain, 640, 480)
m = vn.ok
diff = int(((vo.px[m] != vn.px[m]) | (vo.py[m] != vn.py[m])).sum())
print(f"1. front view: {diff} of {m.sum():,} neurons on a different pixel than before; off-panel at front view: {int((vn.ok & ~vn.vis).sum())}")
if diff or (vn.ok & ~vn.vis).any(): ok_all = False

# 2. a full turn is the identity
px0, py0 = vn.px.copy(), vn.py.copy()
vn.set_view(yaw=2 * np.pi); d360 = int(((vn.px[m] != px0[m]) | (vn.py[m] != py0[m])).sum()); vn.reset_view()
print(f"2. full 360-degree turn: {d360} neurons not back on their start pixel")
if d360 > 5: ok_all = False                                   # a handful may flip a pixel from floating-point rounding

# 3. every firing neuron visible after rotating
b = Behaviour(brain, "escape"); r = brain.run(b.stim, record=b.readouts, seed=1, spikes=True)
fired = np.unique(r["spike_i"])
VIEWS = [("front", 0, 0, 1.0), ("turned 60deg", 60, 0, 1.0), ("from above", 0, 80, 1.0), ("side view", 90, 0, 1.0),
         ("oblique", 45, 30, 1.0), ("zoom x2", 30, 20, 2.0)]
for name, yaw, pitch, zoom in VIEWS:
    vn.set_view(yaw=np.radians(yaw), pitch=np.radians(pitch), zoom=zoom)
    vn.reset(); vn.advance(300.0, r["spike_t"], r["spike_i"])
    img = vn.frame(stimulus=b.stim, readout=b.readout, readouts=b.readouts, title=name, subtitle="",
                   counts=f"{fired.size:,} neurons fired").astype(int)
    base = (np.clip(vn.base, 0, 1) * 255).astype(int)
    on = fired[vn.vis[fired]]; off_screen = int((vn.ok[fired] & ~vn.vis[fired]).sum())
    gain = img[vn.py[on], vn.px[on]].sum(1) - base[vn.py[on], vn.px[on]].sum(1)
    under_text = (vn.py[on] < 70) | (vn.py[on] > vn.H - 50)
    bad = int(((gain < MIN_GAIN) & ~under_text).sum())
    print(f"3. {name:13s} on screen {on.size:4d} | visible {int((gain >= MIN_GAIN).sum()):4d} | invisible {bad} | "
          f"under text {int(under_text.sum())} | off the zoomed panel {off_screen}")
    if bad: ok_all = False
    Image.fromarray(img.astype(np.uint8)).save(os.path.join(OUT, f"{name.replace(' ', '_')}.png"))

# 4. speed
vn.reset_view(); t0 = time.time()
for k in range(20): vn.set_view(yaw=np.radians(3 * k))
t_proj = (time.time() - t0) / 20
t0 = time.time()
for k in range(20): vn.frame(stimulus=b.stim, readout=b.readout, readouts=b.readouts)
t_frame = (time.time() - t0) / 20
print(f"4. re-project for a new view: {1000 * t_proj:.0f} ms | render a frame: {1000 * t_frame:.0f} ms | "
      f"-> {1 / (t_proj + t_frame):.0f} frames/s for the brain panel while rotating (before the 3-D fly is drawn)")
print("PASS" if ok_all else "FAIL")
