"""
LIVE Flappy Bird played by the fruit-fly connectome.   Run:  python live_flappy.py

Keys while it plays:
  F  = fly brain (real wiring)          S = same neurons, wiring SCRAMBLED
  N  = no brain (linear layer only)     C = cut 300 random neurons (press again to cut more)
  R  = restart game                     SPACE = pause     +/- = speed     ESC = quit
Left: the game.  Right: the 3,000 real fly neurons, brightness = activity.
"""
import os, sys, numpy as np, scipy.sparse as sp
import pygame
import flappy_fly as F

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------ brains + read-outs (cached after first run)
def get_brain(kind):
    Wfly = sp.load_npz(os.path.join(HERE, "W_fly.npz"))
    if kind == "fly":
        W = Wfly
    elif kind == "random":
        W = F.scale_spectral(F.shuffle_wiring(sp.load_npz(os.path.join(HERE, "W_fly.npz"))))
    else:
        W = None
    res = F.Reservoir(W, F.Flappy.N_IN) if W is not None else None
    path = os.path.join(HERE, f"readout_{kind}.npy")
    if not os.path.exists(path):
        print(f"training read-out for '{kind}' brain (one-time, ~1 min)...")
        X, Y = F.collect(res, 40000, seed=1); w = F.fit_readout(X, Y); thr = F.calibrate(res, w, X)
        np.save(path, np.append(w, thr))
    wt = np.load(path)
    return res, wt[:-1], wt[-1]

print("loading brains...")
BRAINS = {k: get_brain(k) for k in ("fly", "random", "none")}
ids = np.load(os.path.join(HERE, "fly_ids.npy"))
n = BRAINS["fly"][0].W.shape[0]
LABEL = {"fly": "FLY BRAIN  (real wiring)", "random": "SCRAMBLED WIRING  (same neurons)", "none": "NO BRAIN  (linear layer)"}

# ------------------------------------------------ window
pygame.init()
GW, GH, BW = 520, 560, 520
screen = pygame.display.set_mode((GW + BW, GH)); pygame.display.set_caption("Flappy Bird - played by a fruit-fly brain")
font = pygame.font.SysFont("consolas", 18); big = pygame.font.SysFont("consolas", 26, bold=True); small = pygame.font.SysFont("consolas", 14)
clock = pygame.time.Clock()
rng = np.random.default_rng(0)
ang = rng.uniform(0, 2 * np.pi, n); rad = np.sqrt(rng.uniform(0, 1, n))
NX = (GW + BW / 2 + rad * np.cos(ang) * 230).astype(int); NY = (300 + rad * np.sin(ang) * 230).astype(int)
BG, GREEN, YEL, TXT, DIM, REDC = (11, 15, 23), (61, 220, 151), (255, 217, 102), (230, 237, 247), (139, 152, 173), (255, 92, 108)

mode, cut_mask, fps, paused = "fly", np.ones(n, bool), 60, False
game = F.Flappy(seed=int(rng.integers(1e6))); obs = game.reset()
best = {k: 0 for k in BRAINS}; games_played = {k: 0 for k in BRAINS}; last_scores = {k: [] for k in BRAINS}
flash = 0

def restart():
    global game, obs, flash
    game = F.Flappy(seed=int(rng.integers(1e6))); obs = game.reset()
    res = BRAINS[mode][0]
    if res is not None: res.reset()
    flash = 0

def decide():
    res, w, thr = BRAINS[mode]
    if res is None:
        state = obs
    else:
        state = res.step(obs)
        if mode != "none": res.x *= cut_mask            # cut neurons stay silent
        state = res.x
    return (np.append(state, 1.0) @ w) > thr

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: running = False
            elif e.key == pygame.K_f: mode = "fly"; cut_mask[:] = True; restart()
            elif e.key == pygame.K_s: mode = "random"; cut_mask[:] = True; restart()
            elif e.key == pygame.K_n: mode = "none"; cut_mask[:] = True; restart()
            elif e.key == pygame.K_c and mode != "none":
                alive_idx = np.flatnonzero(cut_mask); cut_mask[rng.choice(alive_idx, min(300, alive_idx.size), replace=False)] = False
            elif e.key == pygame.K_r: restart()
            elif e.key == pygame.K_SPACE: paused = not paused
            elif e.key in (pygame.K_PLUS, pygame.K_EQUALS): fps = min(240, fps + 30)
            elif e.key == pygame.K_MINUS: fps = max(15, fps - 15)

    if not paused:
        flap = decide()
        if flap: flash = 6
        obs, alive = game.step(flap)
        if not alive:
            games_played[mode] += 1; last_scores[mode] = (last_scores[mode] + [game.score])[-10:]
            best[mode] = max(best[mode], game.score)
            pygame.time.wait(600); restart()

    # ---------------- draw game
    screen.fill(BG)
    pygame.draw.rect(screen, (20, 26, 38), (0, 0, GW, GH))
    for px, g in game.pipes:
        x = int(px * GW)
        if -40 < x < GW + 40:
            top = int((g - F.Flappy.GAP / 2) * GH); bot = int((g + F.Flappy.GAP / 2) * GH)
            pygame.draw.rect(screen, GREEN, (x - 16, 0, 32, top)); pygame.draw.rect(screen, GREEN, (x - 16, bot, 32, GH - bot))
    pygame.draw.circle(screen, YEL, (int(F.Flappy.BIRD_X * GW), int(game.y * GH)), 12)
    if flash: pygame.draw.circle(screen, (255, 140, 92), (int(F.Flappy.BIRD_X * GW), int(game.y * GH)), 18, 3); flash -= 1
    screen.blit(big.render(f"pipes: {game.score}", True, TXT), (14, 12))
    screen.blit(font.render(f"best ({mode}): {best[mode]}   games: {games_played[mode]}", True, DIM), (14, 46))
    if last_scores[mode]: screen.blit(font.render(f"last games: {last_scores[mode]}", True, DIM), (14, 68))
    if paused: screen.blit(big.render("PAUSED", True, REDC), (GW // 2 - 50, GH // 2))

    # ---------------- draw brain
    res = BRAINS[mode][0]
    col = TXT if mode == "fly" else (REDC if mode == "random" else DIM)
    screen.blit(big.render(LABEL[mode], True, col), (GW + 16, 12))
    if res is not None:
        act = np.abs(res.x)
        bright = np.clip(act / 0.6, 0, 1)
        for i in range(n):
            if not cut_mask[i]:
                pygame.draw.circle(screen, (70, 20, 30), (NX[i], NY[i]), 2)
            else:
                b = bright[i]; c = (int(40 + 215 * b), int(20 + 120 * b ** 2), int(60 + 40 * (1 - b)))
                pygame.draw.circle(screen, c, (NX[i], NY[i]), 2 + int(3 * b))
        screen.blit(font.render(f"{int(cut_mask.sum()):,} of {n:,} neurons alive   {int((act > 0.3).sum())} active now", True, DIM), (GW + 16, 46))
        screen.blit(small.render(f"real FlyWire IDs, e.g. {ids[0]}  (codex.flywire.ai)", True, DIM), (GW + 16, 68))
    else:
        screen.blit(font.render("no neurons - decision is a straight linear layer on the 34 inputs", True, DIM), (GW + 16, 46))
    y0 = GH - 96
    for line in ("F fly brain   S scrambled   N no brain   C cut 300 neurons",
                 "R restart   SPACE pause   +/- speed   ESC quit",
                 f"speed: {fps} fps    trained part: 3,001 numbers (read-out only)"):
        screen.blit(small.render(line, True, DIM), (GW + 16, y0)); y0 += 22

    pygame.display.flip(); clock.tick(fps)
pygame.quit()
