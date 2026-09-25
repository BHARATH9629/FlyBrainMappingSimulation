"""FLY BRAIN DIGIT READER: a small AI whose hidden layer is a real fruit-fly brain circuit.

Draw a digit with the mouse; the AI reads it through the fly's mushroom body:
  your drawing -> 72 numbers, one per smell channel (glomerulus)
  -> the real FlyWire wiring from those channels to 4,804 Kenyon cells (FIXED: copied from the fly, never trained)
  -> only the 240 most driven Kenyon cells stay on (~5%, as in the fly)
  -> a small trained reader -> the digit.
The model is built by neuroredteam/ml/fly_ai_model.py (run it once).

Keys: mouse = draw, right mouse = erase, C = clear, T = show a test digit it never saw during training,
      F11 = window / full screen, Q = quit
Usage: python virtual_fly/fly_ai.py
"""
import argparse, os, sys, time
import numpy as np, pygame
from PIL import Image, ImageDraw, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "neuroredteam", "ml", "fly_ai_model.npz")
ap = argparse.ArgumentParser(); ap.add_argument("--windowed", action="store_true")
ap.add_argument("--script", default="", help="tests: comma-separated T / C / Q actions, one per second")
ap.add_argument("--shot", default="", help="tests: save the canvas to this PNG just before quitting")
args = ap.parse_args()
if not os.path.exists(MODEL): sys.exit("model missing: run  python neuroredteam/ml/fly_ai_model.py  once first")
m = np.load(MODEL, allow_pickle=False)
PCA_MEAN, PCA_COMP, M, K = m["pca_mean"], m["pca_comp"], m["M"], int(m["k"])
COEF, ICPT, ACC = m["coef"], m["intercept"], float(m["acc"])
KC_POS, KC_IDS = m["kc_pos"], m["kc_ids"]; N_KC = M.shape[1]
GAL_IMG, GAL_LAB, GAL_CODES = m["gal_img"], m["gal_lab"], m["gal_codes"]
TEST_IMG, TEST_LAB = m["test_img"], m["test_lab"]

W, H = 1280, 800
BG, PANEL, INK, DIM, ACCENT = (13, 14, 19), (24, 26, 34), (236, 237, 242), (146, 150, 166), (255, 214, 102)
EDGE, GREEN, BLUE, PINK = (46, 50, 66), (110, 215, 140), (110, 160, 255), (255, 120, 200)
DRAW_R = pygame.Rect(28, 150, 400, 400); PAD = 420            # the drawing pad is drawn at 400 px, kept at 420 internally


# ---------------------------------------------------------------- the AI
def mnistify(canvas):
    """a drawing -> a 28x28 image prepared exactly like MNIST: crop, fit into 20x20, centre by centre of mass"""
    a = np.asarray(canvas, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.1)
    if ys.size < 20: return None
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; h, w = crop.shape; s = 20.0 / max(h, w)
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    small = np.asarray(img, np.float32) / 255.0
    out = np.zeros((28, 28), np.float32); oy, ox = (28 - small.shape[0]) // 2, (28 - small.shape[1]) // 2
    out[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    wy, wx = np.indices(out.shape); tot = out.sum()
    cy, cx = (wy * out).sum() / tot, (wx * out).sum() / tot
    out = np.roll(np.roll(out, int(round(14 - cy)), 0), int(round(14 - cx)), 1)
    return np.clip(out, 0, 1)


def think(img28):
    """the whole AI: returns the 72 channel inputs, the Kenyon cells that fire, and the probability of each digit"""
    z = (img28.ravel() - PCA_MEAN) @ PCA_COMP.T                    # 72 numbers, one per glomerulus
    drive = (z - z.mean()) @ M                                     # through the real fly wiring
    on = np.argpartition(-drive, K - 1)[:K]                        # winner-take-all: the 240 most driven cells
    code = np.zeros(N_KC, np.float32); code[on] = 1.0
    logit = COEF @ code + ICPT; p = np.exp(logit - logit.max()); p /= p.sum()
    return z, on, p, code


# ---------------------------------------------------------------- display (same look as the Phase 3 window)
if sys.platform == "win32":
    try:
        import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception: pass
pygame.init(); pygame.display.set_caption("Fly Brain Digit Reader")


def open_display(full):
    ds = pygame.display.get_desktop_sizes()[0]
    if full:
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"; return pygame.display.set_mode(ds, pygame.NOFRAME)
    os.environ.pop("SDL_VIDEO_WINDOW_POS", None); os.environ["SDL_VIDEO_CENTERED"] = "1"
    k = min((ds[0] - 16) / W, (ds[1] - 110) / H); return pygame.display.set_mode((int(W * k), int(H * k)))


FULL = not (args.windowed or args.script); display = open_display(FULL); screen = pygame.Surface((W, H))


def _fit():
    dw, dh = display.get_size(); k = min(dw / W, dh / H); return k, (dw - int(W * k)) // 2, (dh - int(H * k)) // 2


def present():
    k, ox, oy = _fit()
    if k == 1.0: display.blit(screen, (ox, oy))
    else: display.fill(BG); display.blit(pygame.transform.smoothscale(screen, (int(W * k), int(H * k))), (ox, oy))
    pygame.display.flip()


def to_canvas(pos):
    k, ox, oy = _fit(); return (int((pos[0] - ox) / k), int((pos[1] - oy) / k))


F = lambda size, bold=False: pygame.font.SysFont("segoeui", size, bold=bold)
font, small, cap, head, title_f, huge = F(17), F(14), F(12, True), F(19, True), F(26, True), F(120, True)
mono = pygame.font.SysFont("consolas", 14)


def rrect(c, r, rad=10, w=0): pygame.draw.rect(screen, c, r, width=w, border_radius=rad)


def card(r, title, right="", accent=None):
    rrect((6, 7, 10), r.move(0, 3), 12); rrect(PANEL, r, 12)
    if accent: pygame.draw.rect(screen, accent, (r.x + 14, r.y + 11, 3, 12), border_radius=2)
    screen.blit(cap.render(title.upper(), True, (178, 182, 198)), (r.x + (22 if accent else 14), r.y + 9))
    if right: t = small.render(right, True, DIM); screen.blit(t, (r.right - t.get_width() - 14, r.y + 7))
    return r.y + 32


def pill(right_x, y, text, fg, bg, dot=None):
    t = small.render(text, True, fg); w = t.get_width() + (32 if dot else 20)
    rrect(bg, (right_x - w, y, w, 26), 13)
    if dot: pygame.draw.circle(screen, dot, (right_x - w + 14, y + 13), 4)
    screen.blit(t, (right_x - w + (24 if dot else 10), y + 4)); return right_x - w - 8


def gray_surface(img, size):
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    rgb = np.stack([a, a, a], -1)
    return pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2))), size)


# Kenyon cells at their real positions (front view), inside a faint outline of the whole brain
KC_R = pygame.Rect(470, 330, 380, 300)
BP = m["brain_pts"][:, :2]; lo, hi = np.percentile(BP, 0.5, 0), np.percentile(BP, 99.5, 0)    # robust: ignore stray points
sc = min((KC_R.w - 20) / (hi - lo)[0], (KC_R.h - 20) / (hi - lo)[1]); off = (np.array(KC_R.size) - 20 - (hi - lo) * sc) / 2
def _px(xy):
    out = np.full((len(xy), 2), -100, int); ok = np.isfinite(xy).all(1)
    out[ok] = (np.array(KC_R.topleft) + 10 + off + (xy[ok] - lo) * sc).astype(int); return out
KC_PX, BRAIN_PX = _px(KC_POS[:, :2]), _px(BP)
_bg = pygame.Surface(KC_R.size); _bg.fill((18, 19, 25))
for q in BRAIN_PX - np.array(KC_R.topleft):
    if 0 <= q[0] < KC_R.w and 0 <= q[1] < KC_R.h: _bg.set_at(tuple(q), (46, 54, 78))
for q in KC_PX - np.array(KC_R.topleft):
    if 0 <= q[0] < KC_R.w and 0 <= q[1] < KC_R.h: _bg.set_at(tuple(q), (120, 100, 170))
BRAIN_BG = _bg                                     # drawn once: brain outline (blue-grey) + all Kenyon cells (purple)

pad = Image.new("L", (PAD, PAD), 0); pdraw = ImageDraw.Draw(pad)
state = dict(last=None, img28=None, result=None, truth=None, t_think=0.0)


def recompute():
    img = mnistify(pad.filter(ImageFilter.GaussianBlur(1.2)))
    state["img28"] = img
    if img is None: state["result"] = None; return
    t0 = time.perf_counter(); state["result"] = think(img); state["t_think"] = time.perf_counter() - t0


def show_test_digit():
    i = np.random.randint(len(TEST_IMG)); a = TEST_IMG[i].reshape(28, 28)
    big = Image.fromarray(a).resize((PAD, PAD), Image.NEAREST)
    pad.paste(big); state["truth"] = int(TEST_LAB[i]); recompute()


def draw():
    screen.fill(BG)
    # header
    pygame.draw.circle(screen, ACCENT, (30, 27), 13); pygame.draw.circle(screen, BG, (30, 27), 5)
    screen.blit(title_f.render("Fly Brain Digit Reader", True, INK), (52, 10))
    screen.blit(font.render("an AI whose hidden layer is a real fruit-fly brain circuit", True, DIM), (52 + title_f.size("Fly Brain Digit Reader")[0] + 14, 17))
    xr = pill(W - 12, 13, f"{ACC * 100:.1f}% correct on {int(m['n_test']):,} test digits it never saw", INK, (26, 52, 38), dot=GREEN)
    # the pipeline strip
    steps = [("Your drawing", "28 x 28 pixels"), ("72 smell channels", "one number per glomerulus"),
             ("4,804 Kenyon cells", "real FlyWire wiring, NOT trained"), ("240 cells fire", "the fly's 5% rule"), ("The answer", "small trained reader")]
    x = 12; w = (W - 24 - 4 * 22) // 5
    for i, (a, b_) in enumerate(steps):
        col = ACCENT if i in (2, 3) else EDGE
        rrect((40, 36, 22) if i in (2, 3) else PANEL, (x, 58, w, 44), 12); rrect(col, (x, 58, w, 44), 12, 2 if i in (2, 3) else 1)
        screen.blit(font.render(a, True, INK), (x + 14, 60)); screen.blit(small.render(b_, True, DIM), (x + 14, 80))
        if i < 4: pygame.draw.polygon(screen, (90, 94, 110), [(x + w + 6, 72), (x + w + 16, 80), (x + w + 6, 88)])
        x += w + 22
    # 1. draw
    r = pygame.Rect(12, 114, 432, 620); y = card(r, "1. Draw a digit", right="right-click erases", accent=ACCENT)
    rrect((0, 0, 0), DRAW_R.inflate(4, 4), 8)
    screen.blit(pygame.transform.smoothscale(gray_surface(np.asarray(pad), (PAD, PAD)), DRAW_R.size), DRAW_R.topleft)
    rrect(EDGE, DRAW_R.inflate(4, 4), 8, 2)
    if state["img28"] is None:
        t = head.render("draw 0-9 here with the mouse", True, (110, 114, 130)); screen.blit(t, (DRAW_R.centerx - t.get_width() // 2, DRAW_R.centery - 12))
    screen.blit(cap.render("WHAT THE AI SEES (28 x 28)", True, (150, 156, 176)), (r.x + 16, DRAW_R.bottom + 16))
    if state["img28"] is not None: screen.blit(gray_surface(state["img28"], (112, 112)), (r.x + 16, DRAW_R.bottom + 36))
    else: rrect((18, 19, 25), (r.x + 16, DRAW_R.bottom + 36, 112, 112), 6)
    for k_, tx in enumerate(("C  clear", "T  a test digit it never saw", "F11  full screen / window", "Q  quit")):
        screen.blit(small.render(tx, True, DIM), (r.x + 150, DRAW_R.bottom + 40 + 22 * k_))
    # 2. the fly layer
    r = pygame.Rect(456, 114, 408, 620); y = card(r, "2. Inside the fly's brain", right="mushroom body", accent=ACCENT)
    res = state["result"]
    screen.blit(cap.render("72 SMELL CHANNELS: YOUR DRAWING AS THE FLY'S INPUT", True, (150, 156, 176)), (r.x + 16, y + 4))
    bx, by, bw, bh = r.x + 16, y + 24, r.w - 32, 150
    rrect((18, 19, 25), (bx, by, bw, bh), 6); pygame.draw.line(screen, EDGE, (bx, by + bh // 2), (bx + bw, by + bh // 2))
    if res is not None:
        z = res[0]; zs = z / (np.abs(z).max() + 1e-9); cw = bw / len(z)
        for i, v in enumerate(zs):
            hgt = int(v * (bh // 2 - 6)); x0 = int(bx + i * cw + 1)
            pygame.draw.rect(screen, ACCENT if v > 0 else BLUE, (x0, by + bh // 2 - max(hgt, 0), max(int(cw) - 1, 1), abs(hgt)))
    screen.blit(cap.render(f"4,804 KENYON CELLS AT THEIR REAL POSITIONS IN THE BRAIN", True, (150, 156, 176)), (r.x + 16, KC_R.y - 22))
    screen.blit(BRAIN_BG, KC_R.topleft); rrect(EDGE, KC_R, 6, 1)
    if res is not None:
        for j in res[1]:
            if KC_R.collidepoint(KC_PX[j]): pygame.draw.circle(screen, ACCENT, KC_PX[j], 3)
    msg = (f"{K} of {N_KC:,} cells fire for your drawing ({K / N_KC * 100:.0f}%)" if res is not None else "draw something: the cells that fire light up")
    screen.blit(font.render(msg, True, ACCENT if res is not None else DIM), (r.x + 16, KC_R.bottom + 12))
    screen.blit(small.render("The wiring from the channels to these cells is copied from FlyWire", True, DIM), (r.x + 16, KC_R.bottom + 38))
    screen.blit(small.render("and never changed. Only the last step (the reader) was trained.", True, DIM), (r.x + 16, KC_R.bottom + 56))
    # 3. the answer
    r = pygame.Rect(876, 114, 392, 620); y = card(r, "3. The answer", right=(f"computed in {state['t_think'] * 1000:.1f} ms" if res is not None else ""), accent=GREEN)
    if res is not None:
        p = res[2]; d = int(np.argmax(p))
        t = huge.render(str(d), True, INK); screen.blit(t, (r.x + 24, y - 8))
        screen.blit(head.render(f"{p[d] * 100:.0f}% sure", True, GREEN if p[d] > 0.7 else ACCENT), (r.x + 24 + t.get_width() + 16, y + 40))
        if state["truth"] is not None:
            ok_ = state["truth"] == d
            screen.blit(font.render(f"true answer: {state['truth']}  " + ("- correct" if ok_ else "- wrong"), True, GREEN if ok_ else (255, 110, 100)),
                        (r.x + 24 + t.get_width() + 16, y + 72))
        for i in range(10):
            yy = y + 170 + 26 * i; screen.blit(font.render(str(i), True, INK if i == d else DIM), (r.x + 20, yy - 2))
            rrect((40, 43, 56), (r.x + 44, yy + 3, 250, 14), 7)
            if p[i] > 0.005: rrect(GREEN if i == d else (90, 96, 120), (r.x + 44, yy + 3, max(int(250 * p[i]), 14), 14), 7)
            screen.blit(small.render(f"{p[i] * 100:.0f}%", True, DIM), (r.x + 304, yy))

    else:
        screen.blit(head.render("waiting for a digit ...", True, DIM), (r.x + 24, y + 30))
    yy = r.y + 470
    screen.blit(cap.render("HOW IT WORKS", True, (150, 156, 176)), (r.x + 16, yy)); yy += 22
    for tx in ("Your drawing becomes 72 numbers, one per smell", "channel of the fly. The real fly wiring spreads them",
               "onto 4,804 Kenyon cells; only the 240 most driven", "fire, as in a real fly. A small trained reader turns",
               "that pattern of cells into a digit."):
        screen.blit(small.render(tx, True, DIM), (r.x + 16, yy)); yy += 19
    # footer
    screen.blit(small.render("Fly layer: real FlyWire v783 wiring (uniglomerular projection neurons -> Kenyon cells), used unchanged. "
                             "Trained: only the reader, on 60,000 MNIST digits.", True, DIM), (16, H - 50))
    screen.blit(small.render("Inspired by the fly-olfaction hashing algorithm of Dasgupta, Stevens & Navlakha, Science 2017.", True, DIM), (16, H - 30))
    present()


clock = pygame.time.Clock(); running = True; drawing = None; script = [s_.strip().upper() for s_ in args.script.split(",") if s_.strip()]
t_next = time.time() + 1.5
while running:
    if script and time.time() >= t_next:
        a = script.pop(0); t_next = time.time() + 1.5
        if a == "T": show_test_digit()
        elif a == "C": pad.paste(0, (0, 0, PAD, PAD)); state.update(img28=None, result=None, truth=None)
        elif a == "Q":
            if args.shot: pygame.image.save(screen, args.shot)
            running = False
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_q: running = False
            elif ev.key == pygame.K_c: pad.paste(0, (0, 0, PAD, PAD)); state.update(img28=None, result=None, truth=None)
            elif ev.key == pygame.K_t: show_test_digit()
            elif ev.key == pygame.K_F11: FULL = not FULL; display = open_display(FULL)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and DRAW_R.collidepoint(to_canvas(ev.pos)):
            drawing = ev.button; state["last"] = None; state["truth"] = None
        elif ev.type == pygame.MOUSEBUTTONUP: drawing = None; state["last"] = None
    if drawing:
        mx, my = to_canvas(pygame.mouse.get_pos())
        px, py = (mx - DRAW_R.x) * PAD / DRAW_R.w, (my - DRAW_R.y) * PAD / DRAW_R.h
        colr, rad = (255, 17) if drawing == 1 else (0, 30)
        if state["last"] is not None: pdraw.line([state["last"], (px, py)], fill=colr, width=2 * rad)
        pdraw.ellipse((px - rad, py - rad, px + rad, py + rad), fill=colr); state["last"] = (px, py); recompute()
    draw(); clock.tick(60)
pygame.quit()
