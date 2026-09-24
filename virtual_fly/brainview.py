"""Draw the whole connectome as a dot cloud and animate spikes over it.

BrainView(brain).frame(...) returns an RGB image: every neuron a dim dot, firing neurons glow and fade, the stimulus
neurons cyan, the readout (MN9) green, attacked neurons red, repair neurons blue. Pure numpy + a gaussian blur, so
139k-211k neurons render in ~50 ms per frame.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

TAU_MS = 40.0            # glow decay in brain time
VOXEL_NM = {"female": (4.0, 4.0, 40.0), "male": (8.0, 8.0, 8.0)}   # dataset voxel size (x, y, z) in nm


class BrainView:
    def __init__(self, brain, width=640, height=480, axes=None, margin=0.06):
        self.brain, self.W, self.H = brain, width, height
        # positions in micrometres: FlyWire voxels are 4 x 4 x 40 nm, MaleCNS voxels 8 nm in every direction. The front
        # view only uses the two in-plane axes (equal voxel size), so converting leaves it exactly as it was; rotating
        # without the conversion would squash the brain to a tenth of its real depth.
        P = brain.positions().astype(np.float64) * np.array(VOXEL_NM.get(brain.sex, (1.0, 1.0, 1.0))) / 1000.0
        ok = np.isfinite(P).all(1)
        if axes is None:                         # female FlyWire: x right, y down (frontal view). male CNS: z long axis
            axes = (0, 1) if brain.sex == "female" else (2, 0)
        third = [k for k in range(3) if k not in axes][0]
        self.P3 = P[:, [axes[0], axes[1], third]]                          # screen-x, screen-y, depth
        xy = self.P3[:, :2]
        lo, hi = np.nanpercentile(xy[ok], 0.5, axis=0), np.nanpercentile(xy[ok], 99.5, axis=0)
        span = hi - lo; scale = min((1 - 2 * margin) * width / span[0], (1 - 2 * margin) * height / span[1])
        self.lo, self.scale, self.off = lo, scale, (np.array([width, height]) - span * scale) / 2
        self.center = np.nanmedian(self.P3[ok], axis=0)                     # the point the brain turns around
        self.flip_y = brain.sex == "male"                                   # brain left, nerve cord right, dorsal up
        self.ok = ok
        self.missing = int((~ok).sum())           # neurons with no recorded position: reported on screen, never drawn
        self.yaw = self.pitch = 0.0; self.zoom = 1.0; self._dens_norm = None
        self._project()
        self.glow = np.zeros(brain.N, np.float32)
        self.fired = np.zeros(brain.N, bool)       # every neuron that fired in the last run: stays dimly lit afterwards
        self.last_t = 0.0
        self.font = ImageFont.truetype("arial.ttf", 16) if _has_arial() else ImageFont.load_default()

    # ---- 3-D view -----------------------------------------------------------------------------------------------
    def _project(self):
        """Screen positions for the current rotation and zoom, and the dim background for that view.
        vis = has a position AND lands inside the panel (zooming in pushes some neurons off it; they are hidden, not
        squashed onto the border, and callers can count them)."""
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Q = (self.P3 - self.center) @ (Rx @ Ry).T + self.center
        pix = (Q[:, :2] - self.lo) * self.scale + self.off
        c = np.array([self.W, self.H]) / 2.0; pix = c + (pix - c) * self.zoom
        if self.flip_y: pix[:, 1] = self.H - pix[:, 1]
        inside = np.isfinite(pix).all(1) & (pix[:, 0] >= 0) & (pix[:, 0] <= self.W - 1) & (pix[:, 1] >= 0) & (pix[:, 1] <= self.H - 1)
        self.vis = self.ok & inside
        self.px = np.clip(np.nan_to_num(pix[:, 0], nan=-1), 0, self.W - 1).astype(int)
        self.py = np.clip(np.nan_to_num(pix[:, 1], nan=-1), 0, self.H - 1).astype(int)
        dens = self._accumulate(np.flatnonzero(self.vis), None)
        dens = gaussian_filter(dens, 0.8, truncate=2.5)
        if self._dens_norm is None: self._dens_norm = np.percentile(dens[dens > 0], 97) + 1e-9   # fixed at the front view
        dens = np.clip(dens / self._dens_norm, 0, 1)
        base = np.zeros((self.H, self.W, 3), np.float32); base[..., 0] = 0.10 + 0.22 * dens
        base[..., 1] = 0.12 + 0.26 * dens; base[..., 2] = 0.20 + 0.40 * dens
        self.base = base

    def set_view(self, yaw=None, pitch=None, zoom=None):
        """Rotate (radians) and zoom; the pitch is kept within +-90 degrees so the brain never turns upside down."""
        y = self.yaw if yaw is None else float(yaw); p = self.pitch if pitch is None else float(np.clip(pitch, -np.pi / 2, np.pi / 2))
        z = self.zoom if zoom is None else float(np.clip(zoom, 0.5, 6.0))
        if (y, p, z) != (self.yaw, self.pitch, self.zoom):
            self.yaw, self.pitch, self.zoom = y, p, z; self._project()

    def rotate_by(self, dyaw, dpitch=0.0): self.set_view(yaw=self.yaw + dyaw, pitch=self.pitch + dpitch)
    def zoom_by(self, factor): self.set_view(zoom=self.zoom * factor)
    def reset_view(self): self.set_view(yaw=0.0, pitch=0.0, zoom=1.0)
    def is_front(self): return self.yaw == 0.0 and self.pitch == 0.0 and self.zoom == 1.0


    def reset(self): self.glow[:] = 0; self.fired[:] = False; self.last_t = 0.0

    def snapshot(self, spike_t, spike_i, size, **frame_kw):
        """A small picture of a whole run (the brain as it looks at the end of the 300 ms run), cropped to the brain and
        scaled to `size`; the live animation state is saved and restored, so the main panel is not disturbed."""
        saved = (self.glow.copy(), self.fired.copy(), self.last_t); view = (self.yaw, self.pitch, self.zoom)
        self.reset_view()                               # always the front view, so the pictures can be compared side by side
        self.reset(); self.advance(300.0, spike_t, spike_i); img = self.frame(bare=True, **frame_kw)
        self.glow, self.fired, self.last_t = saved
        xs, ys = self.px[self.vis], self.py[self.vis]; m = 6
        box = (max(int(xs.min()) - m, 0), max(int(ys.min()) - m, 0), min(int(xs.max()) + m, self.W), min(int(ys.max()) + m, self.H))
        out = np.asarray(Image.fromarray(img).crop(box).resize(size, Image.LANCZOS))
        self.set_view(*view)
        return out

    def advance(self, t_ms, spike_t, spike_i):
        """Bring the glow state to brain time t_ms using spikes in (last_t, t_ms]."""
        dt = max(t_ms - self.last_t, 0.0)
        self.glow *= np.float32(np.exp(-dt / TAU_MS))
        m = (spike_t > self.last_t) & (spike_t <= t_ms)
        if m.any(): np.add.at(self.glow, spike_i[m], 1.0); self.fired[spike_i[m]] = True
        self.last_t = t_ms

    _peak = {}

    def _unit_peak(self, size):
        """centre value of a single neuron's blurred dot at this blur size (cached)"""
        if size not in self._peak:
            k = np.zeros((41, 41)); k[20, 20] = 1.0
            self._peak[size] = float(gaussian_filter(k, size, truncate=2.5)[20, 20])
        return self._peak[size]

    _kern = {}

    def _kernel(self, size):
        """a single neuron's blurred dot at this blur size (cached; same filter as gaussian_filter(..., truncate=2.5))"""
        if size not in self._kern:
            r = int(2.5 * size + 0.5); k = np.zeros((2 * r + 3, 2 * r + 3)); k[r + 1, r + 1] = 1.0
            self._kern[size] = gaussian_filter(k, size, truncate=2.5)[1:-1, 1:-1]
        return self._kern[size]

    def _accumulate(self, idx, w):
        """sum of weights per pixel (bincount is far faster than np.add.at)"""
        flat = self.py[idx] * self.W + self.px[idx]
        return np.bincount(flat, weights=w, minlength=self.H * self.W).reshape(self.H, self.W).astype(np.float32)

    def _splat(self, img, idx, color, size=1.0, strength=None):
        """Every neuron in `idx` is drawn as a dot at FULL brightness at its own pixel: the layer is scaled by the peak of
        a single neuron's dot, not by the layer's maximum, so a dense cluster can never dim a neuron firing alone."""
        if len(idx) == 0: return
        size = round(size * 20) / 20               # the blinking ring changes size every frame: keep the dot cache small
        w = None if strength is None else np.clip(strength, 0.0, 1.0).astype(np.float64)
        idx = np.asarray(idx); K = self._kernel(size); k = K.shape[0] // 2
        if idx.size * K.size < 250_000:            # few neurons: add a pre-blurred dot at each one (same result, far cheaper)
            dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
            ys = self.py[idx][:, None] + dy.ravel(); xs = self.px[idx][:, None] + dx.ravel()
            ww = K.ravel()[None, :] * (1.0 if w is None else w[:, None])
            m = (ys >= 0) & (ys < self.H) & (xs >= 0) & (xs < self.W)
            pix, inv = np.unique((ys * self.W + xs)[m], return_inverse=True)     # only the pixels the dots cover
            val = np.clip(np.bincount(inv, weights=np.broadcast_to(ww, m.shape)[m]) / self._unit_peak(size), 0.0, 1.0)
            img.reshape(-1, 3)[pix] += (val[:, None] * np.asarray(color)[None, :]).astype(np.float32)
            return
        layer = gaussian_filter(self._accumulate(idx, w), size, truncate=2.5)
        layer = np.clip(layer / self._unit_peak(size), 0.0, 1.0).astype(np.float32)
        for c in range(3): img[..., c] += layer * color[c]

    def frame(self, stimulus=(), readout=None, attacked=(), repaired=(), readout_hz=None, title="", subtitle="",
              stim_on=False, blink=1.0, stim_label="stimulus neurons", readout_label="readout neuron", show_redblue=True,
              readout_bar=40.0, silenced=(), counts="", readouts=None, score=(), bare=False, chosen=()):
        img = self.base.copy()
        sil = np.array([i for i in silenced if self.vis[i]], int) if show_redblue else np.array([], int)
        if sil.size: self._splat(img, sil, (0.55, 0.06, 0.10), size=1.4)   # fired on the healthy brain, silent now
        was = np.flatnonzero(self.fired & self.vis & (self.glow <= 0.05))          # fired earlier in this run: dim yellow
        if was.size: self._splat(img, was, (0.95, 0.75, 0.25), size=1.7)
        act = np.flatnonzero((self.glow > 0.05) & self.vis)
        if act.size:
            self._splat(img, act, (1.0, 0.85, 0.35), size=1.6, strength=np.minimum(self.glow[act], 3.0))
        stim = np.array([i for i in stimulus if self.vis[i]], int)
        if stim.size: self._splat(img, stim, (0.3, 0.9, 1.0) if stim_on else (0.15, 0.4, 0.5), size=2.2)
        att = np.array([i for i in attacked if self.vis[i]], int)
        if att.size: self._splat(img, att, (1.0, 0.15, 0.1), size=3.0 * blink + 1.0)
        rep = np.array([i for i in repaired if self.vis[i]], int)
        if rep.size: self._splat(img, rep, (0.3, 0.5, 1.0), size=2.5)
        if readout is not None and self.vis[readout]:
            hot = self.glow[readout] > 0.05
            self._splat(img, np.array([readout]), (0.2, 1.0, 0.3) if hot else (0.1, 0.35, 0.15), size=4.5 if hot else 2.5)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)      # rings: 65 % opaque, neurons show through
        A = 165
        # rings on top of everything: the few neurons the teams touched (and those that went silent) must stand out
        # whatever is firing underneath -- a colour splat alone blends into the yellow glow
        if show_redblue:
            for i in sil: x, y = self.px[i], self.py[i]; d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(235, 70, 130, A), width=1)
            for i in rep: x, y = self.px[i], self.py[i]; d.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(90, 150, 255, A), width=2)
            for i in att: x, y = self.px[i], self.py[i]; d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 50, 40, A), width=2)
        # neurons the viewer silenced (YOU TRY IT): a magenta cross on each
        for i in list(chosen)[:800]:
            if self.vis[i]:
                x, y = self.px[i], self.py[i]
                d.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 90, 255, 235), width=2); d.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 90, 255, 235), width=2)
        # the output neuron(s) the behaviour is judged on: always ringed in green -- FILLED if it fired during this run,
        # HOLLOW if it stayed silent -- so "did MN9 fire?" can be read at a glance long after the glow has faded
        outs = [i for i in (readouts if readouts is not None else ([readout] if readout is not None else [])) if self.vis[i]]
        out_fired = False
        for i in outs:
            x, y = self.px[i], self.py[i]; fired_now = bool(self.fired[i] or self.glow[i] > 0.05); out_fired |= fired_now
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(51, 255, 77, A) if fired_now else None, outline=(51, 255, 77, 230), width=2)
        im = Image.alpha_composite(im, ov).convert("RGB")
        if bare: return np.asarray(im)                  # no text: for the small before / after pictures
        d = ImageDraw.Draw(im)
        if outs:                                        # its name and rate, top right: green if it fired in this run
            tag = readout_label.split(" (")[0] + (f"  {readout_hz:.0f} Hz" if readout_hz is not None else "") + ("  (fired)" if out_fired else "  (silent)")
            d.text((self.W - 12 - d.textlength(tag, font=self.font), 8), tag, fill=(51, 255, 77) if out_fired else (170, 190, 170), font=self.font)
        d.text((10, 8), title, fill=(235, 235, 235), font=self.font)
        yt = 28                                                           # text lines under the title, top to bottom
        if subtitle: d.text((10, yt), subtitle, fill=(180, 180, 190), font=self.font); yt += 20
        for txt, col in (score or ()): d.text((10, yt), txt, fill=col, font=self.font); yt += 20    # red / blue scoreboard
        if counts: d.text((10, yt), counts, fill=(255, 217, 89), font=self.font)
        # full legend, left to right
        legend = [((76, 230, 255), stim_label), ((255, 217, 89), "fired in this run"), ((51, 255, 77), readout_label)]
        if show_redblue and not chosen: legend += [((235, 70, 130), "went silent"), ((255, 38, 26), "attacked"), ((77, 128, 255), "repaired")]
        if chosen: legend += [((255, 90, 255), "silenced by you")]
        rows, row, x = [], [], 10                                         # wrap the legend onto as many lines as it needs
        for col, lab in legend:
            w = 15 + d.textlength(lab, font=self.font) + 14
            if row and x + w > self.W - 6: rows.append(row); row, x = [], 10
            row.append((x, col, lab)); x += w
        rows.append(row)
        for r_i, row in enumerate(rows):
            y = self.H - 22 - 20 * (len(rows) - 1 - r_i)
            for x, col, lab in row:
                d.rectangle((x, y + 3, x + 10, y + 13), fill=col); d.text((x + 15, y - 2), lab, fill=(200, 200, 200), font=self.font)
        return np.asarray(im)


def _has_arial():
    import os; return os.path.exists("C:/Windows/Fonts/arial.ttf")
