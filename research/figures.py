"""Paper figures (27 Sep), drawn ONLY from committed result files. Output: results/figures/fig*.png (300 dpi) + .pdf
Palette: reference data-viz palette, slots 1-3 for the three flies (validated all-pairs, light mode); aqua sits below
3:1 contrast on the surface, so every fly is also named by a direct label or legend.
Usage: python research/figures.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from guard import start, finish
out = start("figures")
R = os.path.dirname(out)
FLY = {"FlyWire": "#2a78d6", "BANC": "#eb6834", "male": "#1baf7a"}
LABEL = {"FlyWire": "FlyWire (female)", "BANC": "BANC (female)", "male": "MaleCNS (male)"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "legend.frameon": False, "lines.linewidth": 2})
L = lambda p: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, p)))]
fly = lambda n: n.split("_")[2]


def save(fig, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight"); plt.close(fig); print("wrote", name)


def panel(ax, letter): ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


S = L("instances/s_*_block.json")

# ---------------------------------------------------------------- Fig 2: the screen and its convergence
fz = json.load(open(os.path.join(R, "screen", "frozen.json"))); B = json.load(open(os.path.join(R, "screen", "stage_B.json")))
groups = sorted(fz["groups"].items(), key=lambda kv: -kv[1])
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1, 1.6], wspace=0.45))
steps = [("Input types in\nall three flies", 96), ("Candidate\npathways", len(B)), ("Pass in\n≥ 2 flies", sum(r["n_flies_pass"] >= 2 for r in B)),
         ("Pass in\nall 3 flies", sum(r["n_flies_pass"] == 3 for r in B)), ("Output groups\n(analysis unit)", len(groups))]
y = np.arange(len(steps))[::-1]
a.barh(y, [s[1] for s in steps], height=0.6, color=SEQ[4], edgecolor=SURF, linewidth=2)
for yy, (lab, v) in zip(y, steps): a.text(v + 3, yy, str(v), va="center", color=INK, fontsize=8.5)
a.set_yticks(y, [s[0] for s in steps]); a.set_xlim(0, 200); a.grid(axis="y", visible=False); a.set_xlabel("count")
a.set_title("Systematic screen"); panel(a, "A")
top = groups[:12]; rest = sum(v for _, v in groups[12:])
names = [g.replace("|", " ") for g, _ in top] + [f"{len(groups) - 12} other groups"]; vals = [v for _, v in top] + [rest]
x = np.arange(len(vals))
b.bar(x, vals, width=0.7, color=[SEQ[5] if v >= 10 else SEQ[3] for v in vals[:-1]] + [SEQ[1]], edgecolor=SURF, linewidth=2)
for xx, v in zip(x, vals): b.text(xx, v + 1, str(v), ha="center", color=INK, fontsize=7.5)
b.set_xticks(x, names, rotation=55, ha="right", fontsize=7.5); b.grid(axis="x", visible=False)
b.set_ylabel("included pathways"); b.set_title("Pathways converge on a few outputs"); panel(b, "B")
save(fig, "fig2_screen")

# ---------------------------------------------------------------- Fig 3: attack and repair size (H3)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=0.35))
rng = np.random.default_rng(0)
for k, f in enumerate(FLY):
    rs = [r for r in S if fly(r["instance"]) == f]
    sz = [r["attack"]["size"] for r in rs if r.get("attack")]; nf = sum(1 for r in rs if not r.get("attack"))
    xs = k + rng.uniform(-0.18, 0.18, len(sz))
    a.scatter(xs, sz, s=18, color=FLY[f], edgecolor=SURF, linewidth=0.8, zorder=3)
    a.hlines(np.median(sz), k - 0.3, k + 0.3, color=INK, linewidth=1.5, zorder=4)
    a.text(k, 21.5, f"{len(sz)}/{len(rs)} broken\nmedian {np.median(sz):g}", ha="center", va="bottom", fontsize=7.2, color=INK2)
a.set_xticks(range(3), [LABEL[f].replace(" (", "\n(") for f in FLY]); a.set_ylim(0, 25); a.set_xlim(-0.6, 2.6)
a.grid(axis="x", visible=False); a.set_ylabel("neurons in the smallest attack"); a.set_title("Small attacks break most pathways"); panel(a, "A")
lim = 20
b.plot([0, lim], [0, lim], color=AXIS, linewidth=1, zorder=1); b.text(18.2, 17.0, "repair = attack", ha="center", color=MUTED, fontsize=7, rotation=45, rotation_mode="anchor")
for f in FLY:
    rs = [r for r in S if fly(r["instance"]) == f and r.get("attack") and r.get("repair")]
    ax_ = np.array([r["attack"]["size"] for r in rs], float); re_ = np.array([r["repair"]["size"] for r in rs], float)
    b.scatter(ax_ + rng.uniform(-0.2, 0.2, len(rs)), re_ + rng.uniform(-0.2, 0.2, len(rs)), s=18, color=FLY[f], edgecolor=SURF, linewidth=0.8, label=LABEL[f], zorder=3)
h3 = json.load(open(os.path.join(R, "h3", "h3_screen.json")))["primary"]
b.text(0.0, 1.01, f"repair larger in {h3['repair_larger']} of {h3['repair_larger'] + h3['repair_smaller']} runs (sign test p = {h3['p_one_sided']:.1g})",
       transform=b.transAxes, ha="left", va="bottom", fontsize=7.3, color=INK2)
b.set_xlim(0, lim); b.set_ylim(0, lim); b.set_xticks(range(0, 21, 5)); b.set_yticks(range(0, 21, 5)); b.set_xlabel("attack size (neurons)"); b.set_ylabel("repair size (neurons)")
b.legend(loc="upper left", fontsize=7.2, handletextpad=0.2); b.set_title("Repair costs more than attack (H3)", pad=14); panel(b, "B")
save(fig, "fig3_attack_repair")

# ---------------------------------------------------------------- Fig 4: H1 -- strong drivers predict attack cost
H = json.load(open(os.path.join(R, "h1", "h1_screen.json")))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, f in zip(axs, FLY):
    rows = [r for r in H["rows"] if r["fly"] == f]
    fx = [r for r in rows if r["cost"] < 1000]; nx = [r for r in rows if r["cost"] >= 1000]
    ax.scatter([max(r["R1"], 0.8) for r in fx], [r["cost"] for r in fx], s=18, color=FLY[f], edgecolor=SURF, linewidth=0.8, zorder=3)
    ax.scatter([max(r["R1"], 0.8) for r in nx], [23 + rng.uniform(-0.6, 0.6) for _ in nx], s=18, facecolor="none", edgecolor=FLY[f], linewidth=1.2, zorder=3)
    ax.axhline(21, color=AXIS, linewidth=0.8); ax.set_xscale("log"); ax.set_ylim(0, 25)
    rho = H["report"]["primary"]["R1 vs cost"]["per_fly"][f]
    ax.set_title(LABEL[f], fontsize=8.5); ax.text(0.04, 0.80, f"ρ = {rho:.2f}", transform=ax.transAxes, color=INK2, fontsize=8)
    ax.set_xticks([1, 10, 100], ["1", "10", "100"]); ax.set_xlim(0.6, 250)
fig.supxlabel("strong drivers of the output, R1 (log scale; R1 = 0 drawn at 0.8)", fontsize=8.5, color=INK2, y=-0.04)
axs[0].set_yticks([0, 5, 10, 15, 20, 23], ["0", "5", "10", "15", "20", "none\nfound"]); axs[0].set_ylabel("attack size (neurons)")
pr = H["report"]["primary"]["R1 vs cost"]
fig.suptitle(f"More strong drivers, costlier attack (H1, R1): pooled ρ = {pr['pooled_rho']:.2f}, p = {pr['p_one_sided']:.4f}", x=0.08, ha="left",
             fontsize=9.5, fontweight="bold", y=1.04)
save(fig, "fig4_h1_drivers")

# ---------------------------------------------------------------- Fig 5: scaling sweep
sw = json.load(open(os.path.join(R, "sweep", "sweep_summary.json")))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.35))
for f, pts in (("BANC", [1.0, 1.4, 1.815]), ("male", [0.556, 1.0])):
    xs = pts; ys = [100 * sw[f"{f} x{p}"]["break_rate"] for p in pts]
    a.plot(xs, ys, color=FLY[f], marker="o", markersize=5, markeredgecolor=SURF, label=LABEL[f], zorder=3)
    for xx, yy, p in zip(xs, ys, pts):
        below = (f == "male" and p == 1.0)
        a.text(xx + (0.05 if below else 0), yy + (-8 if below else 3.5), f"{yy:.0f}%", ha="center", fontsize=7.2, color=INK2)
fw = 100 * sw["FlyWire x1.0"]["break_rate"]
a.scatter([1.0], [fw], color=FLY["FlyWire"], s=30, edgecolor=SURF, zorder=4, label=LABEL["FlyWire"]); a.text(1.03, fw + 3.5, f"{fw:.0f}%", fontsize=7.2, color=INK2)
a.axvline(1.0, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3))); a.text(1.02, 3, "raw synapse counts", fontsize=7, color=MUTED)
a.set_xlabel("global weight scale (× synapse count)"); a.set_ylabel("pathways broken (%)"); a.set_ylim(0, 105); a.set_xlim(0.45, 1.95)
a.legend(loc="upper right", fontsize=7.2); a.set_title("Scaling the weights up hides weak spots"); panel(a, "A")


def paired(dir_raw, fly_):
    def load(p): return {os.path.basename(x).split("_block")[0]: json.load(open(x)) for x in glob.glob(os.path.join(R, p))}
    A_, B_ = load(f"{dir_raw}/*.json"), load(f"instances/s_*_{fly_}_block.json")
    both = [k for k in A_ if A_[k].get("included", True) and B_[k].get("included", True)]
    return len(both), sum(bool(A_[k].get("attack")) for k in both), sum(bool(B_[k].get("attack")) for k in both)


nb, rb, sb = paired("sweep_BANC_x1.0", "BANC"); nm, rm, sm = paired("sweep_male_x1.0", "male")
cats = [f"BANC\n{nb} pathways", f"MaleCNS\n{nm} pathways"]; x = np.arange(2); w = 0.36
b1 = b.bar(x - w / 2, [rb, rm], w, color=SEQ[2], edgecolor=SURF, linewidth=2, label="× 1.0 (raw)")
b2 = b.bar(x + w / 2, [sb, sm], w, color=SEQ[5], edgecolor=SURF, linewidth=2, label="pre-registered scale\n(BANC × 1.815, male × 0.556)")
for bars in (b1, b2):
    for r_ in bars: b.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + 0.4, str(int(r_.get_height())), ha="center", fontsize=7.5)
b.set_xticks(x, cats); b.set_ylabel("pathways broken"); b.grid(axis="x", visible=False); b.set_ylim(0, 30); b.set_yticks(range(0, 21, 5))
b.legend(loc="upper left", fontsize=6.8, ncol=2, columnspacing=0.8, handlelength=1.2); b.set_title("Paired: same pathways, two scales"); panel(b, "B")
save(fig, "fig5_scaling")

# ---------------------------------------------------------------- Fig 6: across flies -- kind vs detail (H5)
h5 = json.load(open(os.path.join(R, "h5", "h5_screen.json")))["primary"]; h5s = json.load(open(os.path.join(R, "h5", "h5_screen_strict.json")))["primary"]
tr = json.load(open(os.path.join(R, "h2", "h2_h5_screen_summary.json")))["H5 transfer primary"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.45))
vals = [h5["mean_jaccard"], h5s["null_mean"], h5["null_mean"]]
labs = ["observed", "chance\n(direct inputs)", "chance\n(2-step inputs)"]
bars = a.bar(range(3), vals, width=0.6, color=[SEQ[5], SEQ[1], SEQ[1]], edgecolor=SURF, linewidth=2)
for r_, v in zip(bars, vals): a.text(r_.get_x() + r_.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5)
a.set_xticks(range(3), labs, fontsize=7.3); a.set_ylabel("mean Jaccard overlap of cell types"); a.grid(axis="x", visible=False)
a.set_title(f"Same kinds of weak spot (p = {h5['p_one_sided']:.4f})"); panel(a, "A")
cats = ["copied attack\nbreaks the\nother fly", "does not break"]
b.bar([0, 1], [tr["breaks"], tr["n"] - tr["breaks"]], width=0.6, color=[SEQ[5], SEQ[1]], edgecolor=SURF, linewidth=2)
b.text(0, tr["breaks"] + 1.5, f"{tr['breaks']} ({100 * tr['share']:.0f}%)", ha="center", fontsize=7.5)
b.text(1, tr["n"] - tr["breaks"] + 1.5, f"{tr['n'] - tr['breaks']}", ha="center", fontsize=7.5)
b.axhline(0.75 * tr["n"], color=INK2, linewidth=1, linestyle=(0, (3, 3))); b.text(1.45, 0.75 * tr["n"] + 1.5, "pre-registered bar (75%)", ha="right", fontsize=6.8, color=INK2)
b.set_xticks([0, 1], cats, fontsize=7.3); b.set_ylabel(f"transfers (of {tr['n']})"); b.grid(axis="x", visible=False)
b.set_title("...but not the same exact neurons"); panel(b, "B")
save(fig, "fig6_across_flies")

# ---------------------------------------------------------------- Fig 7: robustness A and C (B added when finished)
RB = L("robust/s_*_block.json")
fig, (a, b, cc) = plt.subplots(1, 3, figsize=(10.2, 2.9), gridspec_kw=dict(wspace=0.5, width_ratios=[1, 1.1, 1.3]))
ok = [r for r in RB if r["attack_grade"] in ("SOLID", "MARGINAL")]; okr = [r for r in RB if r.get("repair_grade") in ("SOLID", "MARGINAL")]
agree = [sum(r["A"]["attack_ok"] >= 12 for r in ok) / len(ok), sum(r["A"].get("repair_ok", 0) >= 12 for r in okr) / len(okr)]
bars = a.bar([0, 1], [100 * v for v in agree], width=0.55, color=SEQ[5], edgecolor=SURF, linewidth=2)
for r_, v, n in zip(bars, agree, (len(ok), len(okr))): a.text(r_.get_x() + r_.get_width() / 2, 100 * v + 2, f"{100 * v:.0f}% of {n}", ha="center", fontsize=7.5)
a.axhline(90, color=INK2, linewidth=1, linestyle=(0, (3, 3))); a.text(0.5, 92, "bar: 90%", ha="center", fontsize=6.8, color=INK2)
a.set_xticks([0, 1], ["attacks", "repairs"]); a.set_ylim(0, 112); a.set_ylabel("still work on the whole brain (%)"); a.grid(axis="x", visible=False)
a.set_title("Whole brain agrees"); panel(a, "A")
c = np.array([r["C"]["share_break"] for r in RB]) * 100
vals_, cnt = np.unique(c.round(0), return_counts=True)
b.bar(vals_, cnt, width=3.2, color=SEQ[3], edgecolor=SURF, linewidth=2)
for v, n in zip(vals_, cnt): b.text(v, n + 0.8, str(n), ha="center", fontsize=7.5, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5), zorder=5)
b.axvline(10, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1); b.text(10.6, max(cnt) * 0.9, "bar: 10%", fontsize=6.8, color=INK2)
b.set_xlabel("random same-size attacks that break it (%)"); b.set_ylabel("pathways"); b.set_xlim(-4, 30); b.grid(axis="x", visible=False)
b.set_title("Random attacks fail"); panel(b, "B")
CB = json.load(open(os.path.join(R, "robust", "control_B.json")))
order = [("W_SYN=0.23375", "weights\n× 0.85"), ("W_SYN=0.31625", "weights\n× 1.15"), ("V_TH=-44", "threshold\n−44 mV"), ("V_TH=-46", "threshold\n−46 mV")]
rh = [CB[k]["rho"] for k, _ in order]
bars = cc.bar(range(4), rh, width=0.6, color=SEQ[5], edgecolor=SURF, linewidth=2)
for i, (r_, (k, _)) in enumerate(zip(bars, order)):
    cc.text(i, rh[i] + 0.03, f"{rh[i]:.2f}\n(n = {CB[k]['n_included']})", ha="center", fontsize=7, color=INK)
cc.axhline(0.7, color=INK2, linewidth=1, linestyle=(0, (3, 3))); cc.text(-0.85, 0.73, "bar: 0.7", ha="left", fontsize=6.8, color=INK2); cc.set_xlim(-0.9, 3.5)
cc.set_xticks(range(4), [l for _, l in order], fontsize=7.3); cc.set_ylim(0, 1.3); cc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
cc.set_ylabel("rank correlation with baseline cost"); cc.grid(axis="x", visible=False)
cc.set_title("Other model settings"); panel(cc, "C")
save(fig, "fig7_robustness")
finish()
