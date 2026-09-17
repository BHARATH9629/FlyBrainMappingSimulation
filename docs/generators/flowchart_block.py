NL = chr(10)
row1 = [("1. Load the brain", "FlyWire data: 139,255 neurons," + NL + "15 M connections   (done)"),
        ("2. Simulator", "spiking neurons; stimulate," + NL + "record, cut, weaken, inject   (done)"),
        ("3. Behaviours + rules", "eat, refuse bitter, groom, escape" + NL + "as pass/fail tests; attack rules"),
        ("4. Atlas + dashboard", "fragility atlas, benchmark," + NL + "Attack / Defend buttons, paper")]
Wb, Hb, GAP, X0, Y1 = 2.75, 1.05, 0.42, 0.55, 1.15
for i, (t, sub) in enumerate(row1):
    x = X0 + i * (Wb + GAP); box(s, x, Y1, Wb, Hb, t, sub, fill=NAVY)
    if i < 3: arrow(s, x + Wb, Y1 + Hb / 2, x + Wb + GAP, Y1 + Hb / 2)
caption(s, 0.55, 2.35, 12.2, 0.4, "How each agent works (attacker and defender run the same loop with opposite goals):", size=13)
loop = [("Choose experiment", "look up wiring, pick neurons" + NL + "to cut, weaken, inject or bias"),
        ("Run the brain", "300 ms simulation with" + NL + "the chosen changes"),
        ("Check behaviour", "MN9 > 60 Hz with sugar" + NL + "and < 10 Hz without"),
        ("Goal met?", "if not, use the result to" + NL + "pick the next one (budget 30)"),
        ("Record answer", "smallest set, neurons," + NL + "simulations used")]
Wl, Hl, GL, XL, YL = 2.25, 1.35, 0.27, 0.55, 2.8
for i, (t, sub) in enumerate(loop):
    x = XL + i * (Wl + GL); box(s, x, YL, Wl, Hl, t, sub, fill=RED if i < 4 else NAVY)
    if i < 4: arrow(s, x + Wl, YL + Hl / 2, x + Wl + GL, YL + Hl / 2)
xg = XL + 3 * (Wl + GL) + Wl / 2; xc = XL + Wl / 2; yb = YL + Hl + 0.35
arrow(s, xg, YL + Hl, xg, yb, head=False); arrow(s, xg, yb, xc, yb, head=False); arrow(s, xc, yb, xc, YL + Hl)
caption(s, 0.55, 4.75, 12.2, 1.0,
        "Attacker goal: break the behaviour with the fewest allowed changes (no motor neurons, realistic rates). "
        "Defender goal: with the attack still in place, restore the behaviour with the fewest stimulated neurons, and it must stay quiet without sugar. "
        "Three versions of each agent share this loop: random choice, greedy ranking, and an LLM that reasons over the wiring. "
        "Every result is repeated over several seeds and checked against published circuits.", size=12)
