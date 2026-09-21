"""Cut a short titled clip from a recorded live session.

Finds the moments where the verdict banner appears (message line changes to '-> VERDICT'), keeps the walk before each
decision at high speed and the decision itself at normal speed, adds title cards, writes one mp4.
Usage: python cut_clip.py --src out/phase2_female_all.mp4 --out out/phase2_60s.mp4
"""
import argparse, numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="out/phase2_female_all.mp4"); ap.add_argument("--out", default="out/phase2_60s.mp4")
ap.add_argument("--fast", type=int, default=6, help="speed-up factor for walking"); ap.add_argument("--hold", type=int, default=50, help="frames kept at 1x after a verdict")
ap.add_argument("--pre", type=int, default=12, help="frames kept at 1x before a verdict")
args = ap.parse_args()

fr = [f for f in iio.get_reader(args.src, "ffmpeg")]; n = len(fr); H, W = fr[0].shape[:2]
MSG = (slice(540, 566), slice(0, 760))                     # the message line region (new UI layout)


def sig(f): return f[MSG].astype(np.int32)


# frames where the message line changes
changes = [0] + [i for i in range(1, n) if np.abs(sig(fr[i]) - sig(fr[i - 1])).mean() > 2.0]
# a verdict = a change where the banner region (top-right of the fly panel) becomes coloured
def has_banner(f):
    reg = f[60:92, 580:632].astype(np.int32); return reg.std(axis=2).mean() > 30     # every banner is right-aligned: this strip is inside it
verdicts = [i for i in changes if has_banner(fr[i]) and not has_banner(fr[max(i - 1, 0)])]
print(f"{n} frames; {len(changes)} message changes; {len(verdicts)} verdict moments at {verdicts}")

font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 44); small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 24)


def card(lines, sub="", frames=45):
    im = Image.new("RGB", (W, H), (16, 17, 22)); d = ImageDraw.Draw(im)
    y = H // 2 - 30 * len(lines)
    for l in lines:
        w = d.textlength(l, font=font); d.text(((W - w) / 2, y), l, fill=(235, 235, 240), font=font); y += 60
    if sub: w = d.textlength(sub, font=small); d.text(((W - w) / 2, y + 10), sub, fill=(150, 152, 165), font=small)
    return [np.asarray(im)] * frames


out = card(["The fly and its brain"], "138,639 real neurons decide what a physics-simulated fly does", 60)
out += card(["No behaviour is scripted.", "Press a key: the stimulus neurons fire, the brain runs, the body acts."], "", 75)
prev = 0
for v in verdicts:
    walk = list(range(prev, max(v - args.pre, prev)))
    out += [fr[i] for i in walk[::args.fast]]                          # fast walk
    out += [fr[i] for i in range(max(v - args.pre, prev), min(v + args.hold, n))]   # the decision at 1x
    prev = min(v + args.hold, n)
out += card(["Six behaviours. One brain.", "Every yellow dot is a neuron that fired."], "Phase 3: what happens when we attack this brain, and whether we can repair it", 90)
iio.mimwrite(args.out, out, fps=30, codec="libx264", quality=8)
print(f"saved {args.out}: {len(out)} frames = {len(out)/30:.0f} s")
