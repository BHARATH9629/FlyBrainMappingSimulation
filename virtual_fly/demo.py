"""Virtual fly demo, split screen: the 3-D fly on the left, its whole connectome on the right.

The fly (NeuroMechFly, MuJoCo) walks to a sugar drop. On arrival the simulated brain is run with the sugar taste
neurons stimulated; every spike is drawn on the dot cloud (slow motion: 300 ms of brain time over the tasting phase)
and the feeding motor neuron MN9 decides whether the proboscis extends. Three brains, straight from the atlas:
    healthy   - nothing changed
    attacked  - the smallest BLOCK attack the red team found (red)
    repaired  - the attack plus the smallest repair the blue team found (blue)

Honest division of labour: walking and steering are FlyGym's controller and a distance rule (nobody has a
connectome-driven nerve cord yet); the eat / no-eat decision, and everything drawn on the brain, is the connectome.

Usage:  python demo.py [--sex female|male] [--modes healthy attacked repaired] [--camera head|side|top] [--out out/]
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as iio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "neuroredteam"))
from brain import Brain
from behaviours import Behaviour
from body import FlyBody
from brainview import BrainView

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"])
ap.add_argument("--modes", nargs="+", default=["healthy", "attacked", "repaired"])
ap.add_argument("--camera", default="head", choices=["head", "side", "top"])
ap.add_argument("--out", default=os.path.join(HERE, "out"))
ap.add_argument("--walk_s", type=float, default=1.8, help="simulated seconds of walking before the sugar")
ap.add_argument("--taste_s", type=float, default=2.0, help="simulated seconds shown after the taste")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--size", type=int, nargs=2, default=[640, 480], help="pixels per panel (w h)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
W, H = args.size


def _iv(d):
    """atlas json -> keyword interventions for Brain.run"""
    out = {}
    for k, v in (d or {}).items():
        if k == "lesion": out[k] = [int(x) for x in v]
        elif k == "potentiate": out[k] = {tuple(int(t) for t in key.strip("()").split(",")): float(f) for key, f in v.items()}
        else: out[k] = {int(a): float(b) for a, b in v.items()}
    return out


def _neurons(iv):
    s = set(iv.get("lesion", [])) | set(iv.get("weaken", {})) | set(iv.get("inject", {})) | set(iv.get("bias", {}))
    s |= {a for (a, b) in iv.get("potentiate", {})}
    return sorted(s)


# ------------------------------------------------------------------ brain: the decision, with every spike recorded
atlas = json.load(open(os.path.join(HERE, "..", "neuroredteam", "atlas", args.sex, "eat.json")))["entry"]
block = atlas["goals"]["block"]
attack = _iv(block["best"]["intervention"])
repair = _iv(block["repair"]["intervention"]) if block.get("repair") else {}
interventions = {"healthy": {}, "attacked": attack, "repaired": {**attack, **repair}}
attack_neurons, repair_neurons = _neurons(attack), _neurons(repair)

print(f"loading the {args.sex} brain ...", flush=True)
brain = Brain(args.sex); beh = Behaviour(brain, "eat")
view = BrainView(brain, W, H)
runs = {}
for mode in args.modes:
    t0 = time.time()
    r_on = brain.run(beh.stim, record=[beh.readout], seed=1, spikes=True, **interventions[mode])
    r_off = brain.run([], record=[beh.readout], seed=1, spikes=True, **interventions[mode])
    on, off = r_on["rates"][beh.readout], r_off["rates"][beh.readout]
    runs[mode] = dict(on_hz=on, off_hz=off, eats=on >= beh.on_thr, spikes_on=(r_on["spike_t"], r_on["spike_i"]),
                      spikes_off=(r_off["spike_t"], r_off["spike_i"]))
    print(f"  {mode:9s}: MN9 {on:6.1f} Hz with sugar, {off:5.1f} Hz without -> {'EATS' if on >= beh.on_thr else 'does NOT eat'}"
          f"   ({r_on['spike_i'].size:,} spikes, {time.time()-t0:.1f}s)", flush=True)

# ------------------------------------------------------------------ body + brain: the show
SUGAR = np.array([12.0, 0.0])
font = ImageFont.truetype("arial.ttf", 18) if os.path.exists("C:/Windows/Fonts/arial.ttf") else ImageFont.load_default()
font_big = ImageFont.truetype("arial.ttf", 28) if os.path.exists("C:/Windows/Fonts/arial.ttf") else font
T_BRAIN = 300.0                       # ms of brain time per run


def hud(frame, lines, banner=None, color=(255, 255, 255)):
    im = Image.fromarray(frame).convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    wmax = max(d.textlength(l, font=font) for l in lines)
    d.rectangle((4, 4, 16 + wmax, 10 + 22 * len(lines)), fill=(255, 255, 255, 190))
    im = Image.alpha_composite(im, ov).convert("RGB"); d = ImageDraw.Draw(im)
    for i, line in enumerate(lines): d.text((10, 8 + 22 * i), line, fill=(20, 20, 20), font=font)
    if banner:
        w = d.textlength(banner, font=font_big); d.rectangle((im.width - w - 24, 8, im.width - 4, 48), fill=color)
        d.text((im.width - w - 14, 13), banner, fill=(0, 0, 0), font=font_big)
    return np.asarray(im)


label = {"healthy": "healthy brain",
         "attacked": f"ATTACKED: {block['best']['strategy'].replace('attacker-', '')} on {block['best']['size']} neurons",
         "repaired": f"REPAIRED: attack + {block['repair']['move']} on {block['repair']['size']}" if block.get("repair") else "REPAIRED"}

for mode in args.modes:
    print(f"rendering {mode} ...", flush=True)
    R = runs[mode]
    body = FlyBody(camera=args.camera, sugar_at=SUGAR, res=(H, W), fps=args.fps)
    n_walk, n_taste = int(args.walk_s / body.timestep), int(args.taste_s / body.timestep)
    at_sugar = False; i_arrive = n_walk
    for i in range(n_walk + n_taste):
        pos = body.position[:2]; d = SUGAR - pos; dist = np.linalg.norm(d)
        if not at_sugar and (dist < 0.95 or i >= n_walk): at_sugar = True; i_arrive = i
        if at_sugar:
            frac = (i - i_arrive) / max(n_taste, 1)
            body.step(0.0, 0.0, eat=1.0 if (R["eats"] and frac > 0.25) else 0.0)     # extend once MN9 has been firing a while
        else:
            err = np.arctan2(d[1], d[0]) - body.heading
            err = (err + np.pi) % (2 * np.pi) - np.pi
            body.step(np.clip(1.0 - 0.8 * err, 0.2, 1.2), np.clip(1.0 + 0.8 * err, 0.2, 1.2), eat=0.0)
    cam = list(body.sim.renderer.frames)[0]
    fly_frames = body.sim.renderer.frames[cam]
    n_render = len(fly_frames); k_arrive = int(round(i_arrive / (n_walk + n_taste) * n_render))
    n_taste_frames = max(n_render - k_arrive, 1)

    # brain panel timeline: walking = the 'no sugar' run played once (mostly dark); tasting = the 'sugar' run in slow motion
    view.reset(); frames = []
    att = attack_neurons if mode != "healthy" else []; rep = repair_neurons if mode == "repaired" else []
    for k, f in enumerate(fly_frames):
        tasting = k >= k_arrive
        if tasting:
            t = T_BRAIN * (k - k_arrive + 1) / n_taste_frames
            if k == k_arrive: view.reset()
            view.advance(t, *R["spikes_on"])
        else:
            t = T_BRAIN * (k + 1) / max(k_arrive, 1)
            view.advance(t, *R["spikes_off"])
        blink = 0.5 + 0.5 * np.sin(k / 3.0) if att else 1.0
        bimg = view.frame(stimulus=beh.stim, readout=beh.readout, attacked=att, repaired=rep,
                          readout_hz=(R["on_hz"] if tasting else R["off_hz"]),
                          title=f"{args.sex} fly connectome: {brain.N:,} neurons, {brain.W.nnz/1e6:.0f} M connections",
                          subtitle=(f"sugar ON   brain time {t:5.0f} ms  (slow motion)" if tasting else f"walking, no sugar   {label[mode]}"),
                          stim_on=tasting, blink=blink)
        lines = [f"{args.sex} fly  |  {label[mode]}", f"sugar neurons: {'ON' if tasting else 'off'}    MN9: {(R['on_hz'] if tasting else R['off_hz']):.0f} Hz"]
        banner = col = None
        if tasting and (k - k_arrive) > 0.25 * n_taste_frames:
            banner = "EATS" if R["eats"] else "NO FEEDING"; col = (120, 230, 120) if R["eats"] else (240, 110, 110)
        fimg = hud(f, lines, banner, col or (255, 255, 255))
        if fimg.shape[:2] != (H, W): fimg = np.asarray(Image.fromarray(fimg).resize((W, H)))
        frames.append(np.concatenate([fimg, bimg], axis=1))
    path = os.path.join(args.out, f"{args.sex}_{mode}.mp4")
    iio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8)
    iio.imwrite(os.path.join(args.out, f"{args.sex}_{mode}_end.png"), frames[-1])
    iio.imwrite(os.path.join(args.out, f"{args.sex}_{mode}_taste.png"), frames[min(k_arrive + n_taste_frames // 6, len(frames) - 1)])
    print(f"  saved {path}  ({len(frames)} frames, arrival at frame {k_arrive})", flush=True)
    body.sim.close(); del body
print("done")
