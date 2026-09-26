# Pre-registration: where behaviour breaks in a whole-brain connectome model

Draft 1, 26 September 2026. Written **before** any of the confirmatory runs below. Changes after this date are
listed at the bottom with their reason; nothing above that list is edited once runs start.

## 1. Question

In a whole-brain spiking model of the fly (FlyWire v783, Shiu et al. 2024 model), which neurons is each behaviour
truly dependent on, what feature of the wiring sets the cost of disabling it, and what does it take to restore it?

## 2. Data sets

| Set | What it contains | Role |
|---|---|---|
| **Discovery** | the 6 current female behaviours (eat, drink, refuse bitter, escape, back away, turn) + hijack, on the sides used so far | ideas came from here; reported, but **not** used to confirm any hypothesis |
| **Confirmation A** | the same behaviours on the **mirror side** of the same brain (stimulus and readout on the other hemisphere), where a mirror version exists | first confirmatory test |
| **Confirmation B** | **new behaviours** added in Stage 1 (inclusion rule below) | second confirmatory test |
| **Confirmation C** | the same behaviours in **BANC** (a second female brain) and, once its data issues are fixed, **MaleCNS** | replication across animals |

**Inclusion rule for any behaviour instance (fixed now):** the healthy model must pass its two-condition test
(readout at or above the ON bar with the stimulus, at or below the OFF bar without it) on at least 8 of 8 check seeds
(1–8), with bars chosen from a published description of the circuit **before** looking at attack results. An instance
that fails is reported and left out; the bars are not adjusted to rescue it.

## 3. Fixed methods (unchanged from Phase 3)

- Attacker and defender: `neuroredteam/agents.py` as of commit `0ea7670`, same rules (`behaviours.RULES`,
  `DEFENCE_RULES`). Screening seeds 1–3, acceptance seeds 1–8. The three core files (`agents.py`, `behaviours.py`,
  `brain.py`) have fingerprint `878888a2`; `research/guard.py` stops any research run if they change, so every
  result is produced by exactly this code.
- Verification: every attack and repair re-checked on **15 unseen seeds (11–25)**, judged per seed:
  SOLID 15/15, MARGINAL 12–14, FAIL below 12.
- Attack cost = size of the smallest verified (SOLID) attack for the goal. Lesion-only cost is also reported
  separately, because it is the quantity graph theory speaks to most directly.
- Repair target: at least 80 % of the healthy rate on every acceptance seed (`Defender.restore(target=0.8)`).

## 4. Hypotheses and tests

**H1. Redundancy at the output predicts attack cost.**
- Measures (computed from the wiring only, never from attack results):
  - *R1, strong drivers*: the number of distinct neurons that give at least 10 excitatory synapses to the readout
    and fire in the healthy stimulus run.
  - *R2, vertex cut*: the smallest number of neurons whose removal disconnects every excitatory path (edges of at
    least 5 synapses, at most 4 steps) from the stimulus neurons to the readout.
- Test: Spearman correlation between attack cost and R1, and between lesion-only cost and R2, across all
  confirmation instances.
- **Confirmed** if both correlations are at least 0.6 with one-sided p < 0.05. **Not confirmed** otherwise, and
  reported as such.

**H2. Attacks hit the output bottleneck; natural suppression does not.**
- Measure: the share of neurons in the cheapest block attack that are direct inputs of the readout.
- For behaviours with a natural suppressor (bitter for feeding, looming for feeding): which neurons that suppressor
  silences or activates, and how many synapses they are from the readout.
- **Confirmed** if, in at least 75 % of confirmation instances, at least half of the attack neurons are direct inputs
  of the readout, while natural suppression acts mainly through neurons at least 2 synapses away.

**H3. Repair is costlier than attack.**
- Test: sign test on (repair size minus attack size) across repairable confirmation instances.
- **Confirmed** if repair is larger in at least 75 % of instances with one-sided p < 0.05.

**H4. Repair restores the output, not the brain (degenerate compensation).**
- (a) Degeneracy: for each attack, look for further repairs that share no neuron with the ones already found (up to
  5). Report how many distinct SOLID repairs exist.
- (b) Non-restoration: distance between network states = 1 minus the correlation of per-neuron spike counts over the
  simulated sub-network. Compare repaired-vs-healthy with healthy-vs-healthy on different seeds.
- **Confirmed** if (a) at least 2 distinct repairs exist in at least half of the instances, and (b) the
  repaired-vs-healthy distance is above the 95th percentile of healthy-vs-healthy distances in at least 75 % of
  instances.

**H5. Weak spots are reproducible across hemispheres and animals.**
- Measure: overlap (Jaccard) of the cell types in matched attacks on the two sides, against a null of random draws of
  the same number of cell types from the readout's direct and second-order inputs.
- Transfer test: apply an attack mapped by cell type to the other side (and to BANC) and check whether it still
  breaks the behaviour.
- **Confirmed** if the overlap beats the null (p < 0.05) and transferred attacks break the behaviour in at least
  75 % of instances.
- Secondary prediction: attacks acting through strong connections (more than 10 synapses) transfer more often than
  attacks through weak ones.

## 5. Controls reported for every result

1. Random attacks of the same size (null model), judged by the same referee.
2. Model-parameter robustness: synapse weight and firing threshold perturbed (`robustness.py`); the ranking of
   attack costs across behaviours must keep a Spearman correlation of at least 0.7 with the baseline.
3. Whole-brain fidelity (`fidelity.py`): results on the 3-step sub-network must match the full brain.

## 6. What will be reported whatever happens

Every hypothesis, confirmed or not; every excluded instance and why; the discovery-set results labelled as discovery.

## 7. Changes after 26 September 2026

(none yet)
