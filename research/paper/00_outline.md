# Paper outline (DRAFT 1, 27 Sep 2026)

**Working title:** *Red-teaming a whole-brain connectome model: small, conserved weak spots and costly repairs across three fly brains*

Target: PLOS Computational Biology (fallbacks: Network Neuroscience, eNeuro, J Comput Neurosci, PLOS ONE);
post a bioRxiv preprint at submission.

## One-paragraph story

Whole-brain connectomes now exist for several flies, and simple spiking models built on them reproduce real
behaviours. We asked how *fragile* these sensory-to-motor pathways are: using an automated attacker and defender on
three independent connectomes (two females, one male), we screened 128 reliable pathways and found that most can be
blocked by silencing or inhibiting about three neurons, almost always next to the output, while restoring them costs
more. How hard a pathway is to break is predicted by how many strong drivers its output has, in all three flies. The
*kinds* of weak spot are conserved across animals — often the same inhibitory cell type — but exact attacks rarely
transfer, and a common cross-data-set normalisation (one global weight scale) can hide weak spots entirely. Results
hold on the whole brain and against random attacks. We list the conserved weak-spot cell types as testable
predictions.

## Sections and headline claims (each tied to a result file)

| # | Section | Claim | Evidence | Figure |
|---|---|---|---|---|
| R1 | Screen | 128 of 164 candidate pathways work reliably in ≥ 2 of 3 connectomes; they converge on 35 outputs (82 on DNb05) | `results/screen/` | Fig 2 |
| R2 | Small attacks | 25/28 FlyWire, 27/32 MaleCNS, 13/27 BANC pathways blocked; median 3 neurons (BANC 5) | `results/instances/s_*` | Fig 3A |
| R3 | Repair (H3) | repair larger than attack in 45/56 runs (p = 3e-6); by output group 21 vs 9 (p = 0.02) | `results/h3/`, `results/audit/` | Fig 3B |
| R4 | Redundancy (H1) | strong-driver count R1 predicts cost in all three (ρ 0.84/0.72/0.78; pooled 0.78, p = 1e-4), also for injection attacks that do not use R1 (ρ 0.64) | `results/h1/`, `results/audit/` | Fig 4 |
| R5 | Where attacks act (H2) | forbidding the output's direct inputs doubles cost or prevents the attack in 50/65 (77 %) | `results/h2/` | (Fig S) |
| R6 | Across animals (H5) | attack cell types overlap far above chance (J 0.25 vs 0.009 strict null); exact transfers break 31 % | `results/h5/` | Fig 6 |
| R7 | Scaling | break rate falls with BANC's weight scale (80 → 61 → 48 %); at × 1 BANC ≈ FlyWire (p = 0.43); paired 16 vs 8 | `results/sweep*/` | Fig 5 |
| R8 | Robustness | whole brain agrees for 95 % of attacks, 98 % of repairs; random same-size attacks break a median 0 %; parameter variants [pending] | `results/robust*/` | Fig 7 |
| R9 | Predictions | 44 conserved weak-spot cell types (10 in all three flies), mostly inhibitory "brakes" | `results/predictions/` | Table 1 |

**Results that are negative or partly negative (need team approval before they go into the text):**
H1 not confirmed as a whole (R2 part fails, ρ 0.23); H2 narrow and not robust to grouping (70 % by group);
H5 transfer not confirmed (31 %); BANC's lower break rate at the pre-registered scale is an artefact; 3 whole-brain
attack mismatches (all BANC × 1.815) and 1 repair mismatch; 7 attacks with no repair; MARGINAL/FAIL grades (11 + 5).

## Figures

1. **Overview** (to draw by hand/diagram tool): connectome → spiking model → attacker/defender loop → verification on
   unseen seeds; three flies.
2. **fig2_screen** — A: screen funnel; B: included pathways per output group (convergence on DNb05).
3. **fig3_attack_repair** — A: smallest attack per fly-run, per connectome (median line); B: repair vs attack size.
4. **fig4_h1_drivers** — attack size vs strong-driver count R1, per connectome; open circles = no attack found.
5. **fig5_scaling** — A: break rate vs global weight scale; B: same pathways at two scales.
6. **fig6_across_flies** — A: cell-type overlap vs two chance levels; B: exact-attack transfers.
7. **fig7_robustness** — A: whole-brain agreement; B: random same-size attacks. (Panel C: parameter variants — pending.)
- **Table 1** — conserved weak-spot cell types (`results/predictions/predictions.md`), with driver lines to be looked up
  in NeuronBridge / Virtual Fly Brain / Meissner et al. 2025 split-GAL4 collection.

## Team split (suggestion)

| Person | Owns | Writes |
|---|---|---|
| 1 | Model, connectomes, scaling | Methods: connectomes, model, scale; Results R7 |
| 2 | Red/blue team, verification | Methods: attack, repair, verification; Results R2, R3, R5 |
| 3 | Screen, pre-registration, statistics | Methods: screen, hypotheses; Results R1, R4, R6, R8 |
| 4 | Figures, predictions, presentation | Fig 1, Table 1 + driver-line lookup, Introduction figure, final layout |
| all | Introduction, Discussion, Abstract | drafted by one, edited by all |

## Still to do before submission

1. Control B result → Fig 7C and R8.
2. Driver lines for Table 1 (manual lookup; do not guess names).
3. Introduction + Discussion (related work: Shiu et al. 2024; Lappalainen et al. 2024; Dorkenwald/Schlegel 2024;
   lesion/robustness studies of connectomes; red-teaming in AI safety as the framing).
4. Check every number in the text against the result files (`research/audit.py` re-derives the main ones).
5. Citations checked; author contributions; AI-use statement; data/code availability.
