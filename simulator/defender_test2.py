"""Defender, attempt 2. Lesson from attempt 1: only neurons that are STILL ACTIVE after the attack can carry the repair.
Candidates = neurons that fire with sugar after the cut AND excite MN9 (ranked by rate x synapses).
Two legal moves: (A) bias them toward threshold; (B) strengthen their synapses onto MN9 (plasticity, <= 3x)."""
import numpy as np, time, scipy.sparse as sp
import sugar_test as st
from defender_test import simulate, report, CUT, mn9, W, N, ids

t0 = time.time()
after = simulate(st.SUGAR, lesion=CUT)                       # who is still active after the attack?
col = W[:, mn9].toarray().ravel() / st.W_SYN                 # synapses onto MN9 (signed)
drive = after * np.clip(col, 0, None)
cands = [int(i) for i in np.argsort(drive)[::-1] if drive[i] > 0 and i not in CUT][:15]
print("Surviving ACTIVE excitatory inputs to MN9 after the attack (rate Hz x synapses):")
for i in cands[:8]: print(f"   {ids[i]}  {after[i]:5.1f} Hz x {int(col[i])} syn")

def simulate_potentiate(stim_ids, lesion, targets, factor, bias=None):
    """Move B: multiply the synapses from `targets` onto MN9 by `factor` (capped), then simulate."""
    global W
    Wm = W.tolil(copy=True)
    for i in targets: Wm[i, mn9] = Wm[i, mn9] * factor
    W_backup, st_W = W, st.W
    import defender_test as dt
    dt.W = Wm.tocsr()
    try: return dt.simulate(stim_ids, lesion=lesion, bias=bias)
    finally: dt.W = W_backup

print("\nMove A: bias the active survivors toward threshold")
found = False
for k in (3, 6, 10):
    for mv in (5.0, 6.5):
        b = {i: mv for i in cands[:k]}
        hs = simulate(st.SUGAR, lesion=CUT, bias=b)[mn9]; hn = simulate([], lesion=CUT, bias=b)[mn9]
        if report(f"bias {k} active survivors by +{mv} mV", hs, hn): found = True; break
    if found: break

print("\nMove B: strengthen the survivors' synapses onto MN9 (plasticity)")
for k in (3, 6):
    for f in (2.0, 3.0):
        hs = simulate_potentiate(st.SUGAR, CUT, cands[:k], f)[mn9]; hn = simulate_potentiate([], CUT, cands[:k], f)[mn9]
        if report(f"x{f:.0f} synapses of {k} active survivors -> MN9", hs, hn): found = True; break
    if found: break

print("\nMove A+B combined" if not found else "")
if not found:
    hs = simulate_potentiate(st.SUGAR, CUT, cands[:6], 2.0, bias={i: 5.0 for i in cands[:6]})[mn9]
    hn = simulate_potentiate([], CUT, cands[:6], 2.0, bias={i: 5.0 for i in cands[:6]})[mn9]
    report("x2 synapses + 5 mV bias, 6 survivors", hs, hn)
print(f"\ntotal time {time.time()-t0:.0f}s")
