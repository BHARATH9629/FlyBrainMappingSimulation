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

- **26 Sep 2026, before any confirmatory run: definition of the mirror instances (Confirmation A).** The 20 sugar
  neurons used by "eat" are a hand-picked left-side set; the female annotation does not separate sugar from water
  taste neurons (64 "LB3 sugar/water" left, 58 right), so "eat" has no exact mirror. Taste is therefore compared as a
  symmetric pair: *taste_L* (all left LB3 → MN9 R) and *taste_R* (all right LB3 → MN9 L); likewise *bitter_L* /
  *bitter_R* (all LB3 + all bitter neurons of one side, MN9 expected quiet). Escape and turn mirror directly
  (*escape_R*: right eye → DNp01 R; *turn_L*: left eye → DNa02 R). Back away is two-sided by design and has no mirror.
  Because the left-side taste and bitter instances overlap the discovery stimuli, they serve only as the reference
  side for H5; **only the right-side / new instances count as confirmation for H1–H4.** Thresholds are the same as
  in the discovery versions (MN9 40 / 10 Hz, DNp01 80 / 10 Hz, DNa02 20 / 10 Hz).
- **26 Sep 2026, after the inclusion check, before any attack run: control for *bitter_R*.** Its control behaviour
  (normal feeding must survive the repair) was *taste_R*, which failed inclusion (35 Hz). Replaced by left taste →
  MN9 L: the same output neuron as *bitter_R*, and it must itself pass inclusion. Inclusion results: *taste_L*,
  *bitter_R*, *escape_R* included; *taste_R* (1/8), *bitter_L* (4/8), *turn_L* (1/8) excluded.
- **26 Sep 2026, before its inclusion check: new behaviour, antennal grooming (Confirmation B).** *groom_L*: the
  Johnston's-organ touch neurons of Shiu et al. 2024 (their JO-CE, JO-F and JO-m lists from `figures.ipynb`; 145 of
  146 present in v783, all left) → aDN1 (DNg62), the neuron they used for grooming. *groom_R*: all right-side
  neurons of the same cell types → the other aDN1 (DNg62 L). Thresholds 40 / 10 Hz, as in the pre-existing grooming
  spec. Deviation from the paper: stimuli are driven at 150 Hz like every other behaviour here (the paper used up to
  220 Hz).
- **26 Sep 2026, before any BANC behaviour run: BANC (Confirmation C).** Loader `research/banc.py` (same sign rule:
  GABA and glutamate inhibitory; glia left out; FlyWire-matched cell types). **Synapse scaling:** across 3,448 cell
  types present with the same neuron count in both flies, FlyWire neurons receive a median 1.815× more input synapses
  than their BANC counterparts (IQR 1.29–2.55), so the model (tuned on FlyWire counts) would be under-driven in BANC for
  reasons of synapse detection, not wiring. **Primary analysis: all BANC weights × 1.815** (computed from wiring only,
  before any behaviour run); the unscaled version is reported as a secondary check. Behaviours are defined by cell
  type: eat (sugar LB3b + LB3c), drink (water LB3a), bitter (sugar + LB1a–e, MN9 expected quiet, control = eat of the
  same side), escape, back away, turn (same types and thresholds as FlyWire). BANC splits taste neurons into subtypes
  that FlyWire lumps together, and the side labels of taste neurons look uneven, so the MN9 readout is **both MN9s
  averaged** (no side chosen). The same inclusion rule applies. All BANC instances count as confirmation.
- **26 Sep 2026, after the BANC inclusion check, before any BANC attack run.** Included (primary, scaled):
  *b_bitter_R*, *b_escape_R*, *b_retreat*, *b_turn_R*, *b_turn_L*; excluded: *b_eat_L/R*, *b_drink_L/R*,
  *b_bitter_L*, *b_escape_L*. Because *b_eat_R* (the control of *b_bitter_R*) was excluded, no repair of *b_bitter_R*
  could pass its control check; *b_bitter_R* therefore counts for the attack hypotheses (H1, H2) only, not for the
  repair hypotheses (H3, H4). The same applies to the FlyWire *bitter_R* instance only if its control fails for the
  same reason (it did not: its control passed inclusion).
- **26 Sep 2026, before running H5: exact H5 procedure.** Attacks compared are the *best* attack of each instance (the
  one the verification used; discovery ones from `neuroredteam/atlas/female/<behaviour>.json`). Pairs: escape
  FlyWire L ↔ R; turn FlyWire (right eye) ↔ BANC turn right eye; BANC turn right eye ↔ left eye; bitter FlyWire
  L (discovery) ↔ R. **Overlap test:** Jaccard index of the two attacks' cell-type sets; null = 10,000 draws of the same
  numbers of cell types from each output's direct + second-step input neurons in its own brain; one-sided p = share of
  draws with Jaccard at least as large. **Transfer test:** every attacked neuron is replaced by all neurons of the same
  cell type on the target side (mirror side within a fly; same side across flies), same move and strength; cell types
  missing in the target are dropped and counted. A transfer "breaks" the behaviour if its goal is met on at least 12 of
  the 15 unseen seeds (11–25). Transfers are run in both directions for every pair whose two instances passed inclusion.
- **26 Sep 2026, after seeing H2's first measure, before running this test: H2 revised.** The first H2 measure (share
  of attack neurons that are direct inputs of the output) is partly built into the attacker: its "jam inhibitors"
  strategy only considers direct inhibitory inputs. It is therefore reported but **not used to judge H2**. H2 is judged
  instead by a threat-model test: the same three strategies are rerun with **every direct presynaptic partner of the
  output forbidden** (the agents then fall back one step upstream), and the best attack is checked on the 15 unseen
  seeds (found = goal met on at least 12 of 15, the same standard as the baseline attacks it is compared with).
  **H2 is confirmed** if, in at least 75 % of the confirmation instances with a baseline attack (*escape_R*,
  *bitter_R*, *b_turn_R*, *b_turn_L*), the forbidden-direct-inputs attack costs at least twice the baseline or is not
  found within the rules. Discovery instances are run the same way and reported as discovery.
- **26 Sep 2026, before running it: secondary analysis of BANC's unbreakable behaviours.** (a) "How close": in the
  primary (scaled) BANC, apply the strongest attacks the rules allow (the 20 strongest inhibitors of the output jammed at
  150 Hz; separately, the 10 strongest active drivers cut) to *b_escape_R* and *b_retreat*, and report the output rate
  on the 15 unseen seeds against the 10 Hz OFF bar. (b) Unscaled BANC (weights × 1, the pre-registered secondary
  model): the full attack, repair and verification pipeline on *b_escape_L* and *b_escape_R* (both pass inclusion
  unscaled). Both are reported as secondary; they do not change the primary results.
