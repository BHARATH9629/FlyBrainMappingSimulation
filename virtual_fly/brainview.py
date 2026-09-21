"""Draw the whole connectome as a dot cloud and animate spikes over it.

BrainView(brain).frame(...) returns an RGB image: every neuron a dim dot, firing neurons glow and fade, the stimulus
neurons cyan, the readout (MN9) green, attacked neurons red, repair neurons blue. Pure numpy + a gaussian blur, so
139k-211k neurons render in ~50 ms per frame.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

TAU_MS = 25.0            # glow decay in brain time


class BrainView:
    def __init__(self, brain, width=640, height=480, axes=None, margin=0.06):
        self.brain, self.W, self.H = brain, width, height
        P = brain.positions().astype(np.float64)
        ok = np.isfinite(P).all(1)
        if axes is None:                         # female FlyWire: x right, y down (frontal view). male CNS: z long axis
            axes = (0, 1) if brain.sex == "female" else (2, 0)
        xy = P[:, list(axes)]
        lo, hi = np.nanpercentile(xy[ok], 0.5, axis=0), np.nanpercentile(xy[ok], 99.5, axis=0)
        span = hi - lo; scale = min((1 - 2 * margin) * width / span[0], (1 - 2 * margin) * height / span[1])
        off = (np.array([width, height]) - span * scale) / 2
        pix = (xy - lo) * scale + off
        if brain.sex == "male": pix[:, 1] = height - pix[:, 1]    # brain left, nerve cord right, dorsal up
        self.px = np.clip(np.nan_to_num(pix[:, 0], nan=-1), 0, width - 1).astype(int)
        self.py = np.clip(np.nan_to_num(pix[:, 1], nan=-1), 0, height - 1).astype(int)
        self.ok = ok
        # static background: density of all neurons, dim blue-grey
        dens = np.zeros((height, width)); np.add.at(dens, (self.py[ok], self.px[ok]), 1.0)
        dens = gaussian_filter(dens, 0.8); dens = dens / (np.percentile(dens[dens > 0], 97) + 1e-9)
        base = np.zeros((height, width, 3)); base[..., 0] = 0.10 + 0.22 * np.clip(dens, 0, 1)
        base[..., 1] = 0.12 + 0.26 * np.clip(dens, 0, 1); base[..., 2] = 0.20 + 0.40 * np.clip(dens, 0, 1)
        self.base = base
        self.glow = np.zeros(brain.N, np.float32)
        self.last_t = 0.0
        self.font = ImageFont.truetype("arial.ttf", 16) if _has_arial() else ImageFont.load_default()

    def reset(self): self.glow[:] = 0; self.last_t = 0.0

    def advance(self, t_ms, spike_t, spike_i):
        """Bring the glow state to brain time t_ms using spikes in (last_t, t_ms]."""
        dt = max(t_ms - self.last_t, 0.0)
        self.glow *= np.float32(np.exp(-dt / TAU_MS))
        m = (spike_t > self.last_t) & (spike_t <= t_ms)
        if m.any(): np.add.at(self.glow, spike_i[m], 1.0)
        self.last_t = t_ms

    def _splat(self, img, idx, color, size=1.0, strength=None):
        if len(idx) == 0: return
        layer = np.zeros((self.H, self.W))
        w = np.ones(len(idx)) if strength is None else strength
        np.add.at(layer, (self.py[idx], self.px[idx]), w)
        layer = gaussian_filter(layer, size)
        layer = layer / (layer.max() + 1e-9)
        for c in range(3): img[..., c] += layer * color[c]

    def frame(self, stimulus=(), readout=None, attacked=(), repaired=(), readout_hz=None, title="", subtitle="",
              stim_on=False, blink=1.0):
        img = self.base.copy()
        act = np.flatnonzero((self.glow > 0.05) & self.ok)
        if act.size:
            self._splat(img, act, (1.0, 0.85, 0.35), size=1.6, strength=np.minimum(self.glow[act], 3.0))
        stim = np.array([i for i in stimulus if self.ok[i]], int)
        if stim.size: self._splat(img, stim, (0.3, 0.9, 1.0) if stim_on else (0.15, 0.4, 0.5), size=2.2)
        att = np.array([i for i in attacked if self.ok[i]], int)
        if att.size: self._splat(img, att, (1.0, 0.15, 0.1), size=3.0 * blink + 1.0)
        rep = np.array([i for i in repaired if self.ok[i]], int)
        if rep.size: self._splat(img, rep, (0.3, 0.5, 1.0), size=2.5)
        if readout is not None and self.ok[readout]:
            hot = self.glow[readout] > 0.05
            self._splat(img, np.array([readout]), (0.2, 1.0, 0.3) if hot else (0.1, 0.35, 0.15), size=4.5 if hot else 2.5)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((10, 8), title, fill=(235, 235, 235), font=self.font)
        d.text((10, 28), subtitle, fill=(180, 180, 190), font=self.font)
        # full legend, left to right
        x = 10; y = self.H - 22
        for col, lab in ((( 76, 230, 255), "sugar neurons"), ((255, 217, 89), "firing"), ((51, 255, 77), "MN9 feeding neuron"),
                         ((255, 38, 26), "attacked"), ((77, 128, 255), "repaired")):
            d.rectangle((x, y + 3, x + 10, y + 13), fill=col); d.text((x + 15, y - 2), lab, fill=(200, 200, 200), font=self.font)
            x += 15 + d.textlength(lab, font=self.font) + 14
        if readout_hz is not None:
            d.text((self.W - 150, 8), f"MN9  {readout_hz:5.0f} Hz", fill=(51, 255, 77) if readout_hz >= 40 else (255, 120, 110), font=self.font)
        return np.asarray(im)


def _has_arial():
    import os; return os.path.exists("C:/Windows/Fonts/arial.ttf")
