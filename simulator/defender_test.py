"""Feasibility test for the DEFENDER.
Attack: cut the 3 command neurons that drive MN9 (feeding stops).
Defence: we may not undo the cut. We may only 'bias' up to K other neurons, i.e. hold them a few mV closer to
threshold (what a tonic stimulation implant does). A repair is valid only if
   (a) with sugar  MN9 fires again  (> 40 Hz)      and
   (b) without sugar MN9 stays quiet (< 10 Hz)     -> not simply forcing feeding."""
import numpy as np, time
import sugar_test as st

ids, id2idx, W, N = st.ids, st.id2idx, st.W, st.N
mn9 = id2idx[st.MN9]
CUT = [id2idx[i] for i in (720575940623211725, 720575940627099338, 720575940632252743)]

def simulate(stim_ids, lesion=(), bias=None, seed=1, t_run=600.0):
    """bias: dict {neuron index: mV added to resting potential} (kept below threshold -> no spontaneous firing)."""
    rng = np.random.default_rng(seed)
    stim = np.array([id2idx[i] for i in stim_ids if i in id2idx], dtype=int)
    Wl = W
    if lesion:
        Wl = W.tolil(copy=True); Wl[list(lesion), :] = 0; Wl = Wl.tocsr()
    v0 = np.full(N, st.V0, np.float32)
    if bias:
        for i, mv in bias.items(): v0[i] = min(st.V_TH - 0.5, st.V0 + mv)
    v = v0.copy(); g = np.zeros(N, np.float32); refr = np.zeros(N, np.float32)
    rfc = np.full(N, st.T_RFC, np.float32); rfc[stim] = 0
    n_steps = int(t_run / st.DT); n_delay = int(round(st.T_DLY / st.DT))
    queue = [np.empty(0, np.int64) for _ in range(n_delay)]; counts = np.zeros(N, np.int64)
    p_poi = st.R_POI * st.DT / 1000; w_poi = st.W_SYN * st.F_POI; dv = st.DT / st.T_MBR; dg = np.float32(np.exp(-st.DT / st.TAU))
    for step in range(n_steps):
        arriving = queue.pop(0)
        if arriving.size: g += np.asarray(Wl[arriving].sum(axis=0)).ravel()
        if stim.size:
            hit = stim[rng.random(stim.size) < p_poi]; v[hit] += w_poi
        act = refr <= 0
        v[act] += dv * (v0[act] - v[act] + g[act]); g[act] *= dg; refr -= st.DT
        spk = np.flatnonzero(v > st.V_TH)
        if spk.size:
            v[spk] = st.V_RST; g[spk] = 0; refr[spk] = rfc[spk]; counts[spk] += 1
        queue.append(spk)
    return counts * (1000.0 / t_run)      # Hz

def report(label, hz_sugar, hz_none):
    ok = hz_sugar > 40 and hz_none < 10
    print(f"  {label:48s} MN9 with sugar = {hz_sugar:5.1f} Hz | without sugar = {hz_none:5.1f} Hz  -> {'VALID REPAIR' if ok else 'fail'}")
    return ok

t0 = time.time()
print("Baseline (healthy):")
report("intact", simulate(st.SUGAR)[mn9], simulate([])[mn9])
print("Attack (3 command neurons cut):")
report("cut top-3", simulate(st.SUGAR, lesion=CUT)[mn9], simulate([], lesion=CUT)[mn9])

# candidates for repair: the remaining excitatory inputs to MN9 (ranked by synapse count), excluding the cut ones
col = W[:, mn9].toarray().ravel() / st.W_SYN
cands = [i for i in np.argsort(col)[::-1] if col[i] > 0 and i not in CUT][:12]
print(f"\nDefender candidates (surviving inputs to MN9): {[(int(col[i])) for i in cands]} synapses each")

print("\nDefence attempts (bias = hold neurons a few mV closer to threshold, never above it):")
for k in (2, 4, 6):
    for mv in (3.0, 5.0):
        bias = {i: mv for i in cands[:k]}
        hs = simulate(st.SUGAR, lesion=CUT, bias=bias)[mn9]; hn = simulate([], lesion=CUT, bias=bias)[mn9]
        if report(f"bias top-{k} surviving inputs by +{mv:.0f} mV", hs, hn):
            print(f"\n  -> repair found: {k} neurons, +{mv:.0f} mV, after {time.time()-t0:.0f}s"); break
    else:
        continue
    break
print(f"\ntotal time {time.time()-t0:.0f}s")
