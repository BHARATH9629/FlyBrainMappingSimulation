# NeuroRedTeam

**Attack and repair testing of a real fly brain.** Two flies, kept strictly apart: the **female** brain (FlyWire, 2024) and the **male** brain + nerve cord (MaleCNS, Google/Janelia, Sep 2026).
RVCE Experiential Learning, Semester 3 (Sept to Dec 2026).
Team: Govind, Bharath, Dhanush A, Shreenidhi Chalmi.

## What this is, in one paragraph

Scientists mapped every neuron and connection in a fruit fly's brain (FlyWire, Nature 2024: 139,255 neurons,
15 million connections) and released it free. A follow-up paper showed the whole map can be simulated on a laptop
and behaves like a real fly. We run that simulation and treat the brain like a system under security testing:
an **attacker agent** searches for the smallest set of allowed changes that breaks a behaviour, a **defender agent**
searches for the smallest stimulation that restores it, and we record the results for every behaviour we can
measure. The output is a **fragility atlas** of fly behaviours and a benchmark for the agents.

## Status (18 September 2026)

| Piece | Status | Evidence |
|---|---|---|
| Whole-brain simulation, matches Nature 2024 | done | sugar -> feeding neuron MN9 fires (95 to 107 Hz); random input -> 0 Hz |
| Behaviour "eat" as a pass/fail test | done | on >= 40 Hz with sugar AND off <= 10 Hz without |
| Attack rules (attack surface) | done | `neuroredteam/behaviours.py` |
| Attacker: greedy (cut), inject-inhibitors (no cuts), weaken-only, random baseline | **done, verified on 3 seeds x 3 seed-sets** | smallest block = 2 neurons via inhibition (beats the hand-found 3); force = 1 neuron |
| Defender: greedy, escalating moves, set minimisation | **done, verified on 3 seeds x 3 seed-sets** | 3-neuron repair after the inhibitory attack; 52 % recovery; 0 Hz without sugar |
| Fragility-atlas entry (eat) with redundancy and force goals | done | block 2, force 1, 1 independent cut-attack, repair 3; zero variance across seed-sets |
| Batch runner + atlas CSV with mean +/- sd | done | `neuroredteam/batch.py` |
| **Male fly** (MaleCNS v1.0, 211,577 neurons, 26 M connections): same simulator, same behaviour test, same agents | **done, 1 run** | sugar -> MN9 143 Hz, 0 Hz without; block 4, force 1, repair 1 |
| Goal "corrupt" (eat bitter) | not yet | needs bitter neuron IDs |
| Behaviours escape, refuse_bitter (+ corrupt goal), both sexes | **done** | milestone 3 below; groom / courtship left out with numbers |
| Search attacker, choke-point maps, cross-sex matching, iterated attack/repair game | **done** | milestone 3 below |
| LLM attacker and defender | not yet | same tools; needs an API key |
| Fragility atlas over all behaviours | 3 behaviours x 2 sexes | milestone 3 |
| Virtual fly (3-D NeuroMechFly body, connectome decides eat/no-eat) | **done, v1** | `virtual_fly/demo.py`; 6 clips: female/male x healthy/attacked/repaired |
| Web dashboard | not yet | |
| Side result: 3,000 fly neurons play Flappy Bird | done | `flappy/`; real wiring did NOT beat scrambled wiring, we say so |

## Quick start

```bash
pip install numpy scipy pandas pyarrow matplotlib pygame-ce
python data/get_data.py            # female connectome, 100 MB -> data/female_flywire/  (not stored in the repo)
python data/get_data.py --male     # male connectome, 1.1 GB -> data/male_malecns/     (optional)

cd neuroredteam
python run_redteam.py --sex female # block + force attacks, redundancy, repair; ~8 min; writes atlas/female/eat.json
python run_redteam.py --sex male   # same on the male fly; ~11 min; writes atlas/male/eat.json
python batch.py --sex female --reps 3   # the atlas table with mean +/- sd over independent seed-sets; ~10 min
```

Other things to run:

```bash
cd simulator && python sugar_test.py        # the Nature 2024 reproduction (about 80 s)
cd simulator && python lesion_test.py       # trace tongue -> MN9 (3 synapses) and cut neurons
cd flappy && python live_flappy.py          # live Flappy Bird played by 3,000 fly neurons (keys: F S N C R)
```

Everything runs on a normal laptop, no GPU.

## Folder map

| Folder | Contents |
|---|---|
| `neuroredteam/` | **the project.** `brain.py` fast simulator + interventions + wiring queries; `behaviours.py` behaviour tests and rules; `agents.py` attacker and defender; `run_redteam.py` one-command run; `atlas/female/` and `atlas/male/` results and full experiment logs, never mixed |
| `simulator/` | the original whole-brain reproduction (`sugar_test.py`), lesion tracing, animations, defender feasibility tests |
| `flappy/` | the Flappy Bird side result: `flappy_fly.py` (train + compare), `live_flappy.py` (live game), `tune.py` |
| `data/` | `get_data.py` downloads the female files into `female_flywire/` and (with `--male`) the male files into `male_malecns/` (all gitignored, 100 MB + 1.1 GB) |
| `results/`, `animations/` | charts, GIFs, the methodology flowchart PNG |
| `virtual_fly/` | 3-D fly (FlyGym/MuJoCo) driven by the connectome's decision: walks to sugar, eats or not depending on healthy / attacked / repaired brain; `out/` has the clips |
| `docs/` | `NeuroRedTeam_Phase1.pptx` (also at repo root), explainer PDF, and the scripts that generate them |
| `HANDOFF.md` | how the code works, who owns what, what to do next, known caveats |

## The fragility atlas so far

### Female fly (FlyWire v783)

Every number verified on 3 seeds per run, repeated over 3 independent seed-sets (`neuroredteam/atlas/female/atlas.csv`).

| Behaviour | Healthy | Smallest BLOCK | Smallest FORCE | Independent cut-attacks | Smallest REPAIR | Recovery |
|---|---|---|---|---|---|---|
| eat | 98.9 +/- 6.0 Hz | **2 neurons, no cuts** (drive two of MN9's own inhibitory inputs: 720575940636809646, 720575940620156209); or cut 3 command cells (720575940623211725, 720575940627099338, 720575940632252743) | **1 neuron** (drive command cell 720575940627099338 -> 94 Hz with no sugar) | 1 (after the 3 command cells, cutting 10 more drivers only reaches 40 Hz) | **3 neurons**, bias +6 mV and x2 synapses onto MN9 | 52 +/- 9 % |

### Male fly (MaleCNS v1.0) -- 3 seeds per run, 3 independent seed-sets (`neuroredteam/atlas/male/atlas.csv`)

Stimulus = left labellar sugar GRNs (types LB3b + LB3c, 17 neurons); readout = left MN9. Neurons named by MaleCNS cell type.

| Behaviour | Healthy | Smallest BLOCK | Smallest FORCE | Independent cut-attacks | Smallest REPAIR | Recovery |
|---|---|---|---|---|---|---|
| eat | 144 +/- 1 Hz | **4.7 +/- 0.5 cuts** (seed-set 1: DNge062, DNge080, GNG120, GNG108); or drive 7 GABA inhibitors | **1 neuron** (drive DNge062 -> 89 Hz with no sugar) | 1.0 +/- 0 | **4.7 +/- 3.9** (1, 3, 10; potentiate only) | 36 +/- 5 % |

Female vs male, in the model: the male feeding circuit needs more damage to block (4 to 5 cuts vs 3; 7 inhibitors vs 2),
is equally easy to hijack (1 command neuron in both), and its repair is seed-sensitive (1 to 10 neurons) where the
female's is not (always 3). Different animal, different lab, different proofreading, so this is a model prediction,
not a biological claim. Still to do: right-side and bilateral stimulus, and whether the attacked *cell types* match
across sexes.

### Checks a reviewer will ask for (done)

**Full-brain fidelity (male).** The atlas runs on the sub-network reachable from the stimulus (46k neurons). We re-ran the
healthy, attacked and repaired brains on the whole reachable CNS (185k neurons, 5 s per run): healthy 166 vs 143 Hz,
attacked 5.6 vs 8.9 Hz, repaired 71 vs 63 Hz, every condition on the same side of the 40 / 10 Hz thresholds
(`neuroredteam/atlas/male/fidelity_eat.json`). The female sub-network was checked the same way in milestone 2.

**Threshold sensitivity.** Rerunning attacker and defender with ON/OFF cut-offs of 30/15, 40/10, 50/10 and 60/5 Hz
(`neuroredteam/atlas/*/sensitivity_eat.csv`):

| on/off Hz | female block | female force | female repair | male block | male force | male repair |
|---|---|---|---|---|---|---|
| 30/15 | 2 | 1 | 3 | 4 | 1 | 1 |
| 40/10 (atlas) | 2 | 1 | 3 | 4 | 1 | 1 |
| 50/10 | 2 | 1 | 4 | 4 | 1 | 2 |
| 60/5 | 2 | 1 | 5 | 5 | 1 | none within budget |

Attack sizes do not depend on the threshold; repair size grows with a stricter ON threshold, as it should.
While doing this we found and fixed a defender bug: it returned the first plan that worked rather than the smallest
(at 30/15 that gave a 10-neuron repair where a 3-neuron one existed). All atlas numbers above were regenerated after the fix.

What this says about feeding (female): robust to graded damage (weakening 10 drivers to 50 % does not block it), fragile to targeted
silencing (3 cells) and to its own inhibition (2 cells), and far easier to hijack (1 cell) than to stop. The attacker found the
2-neuron inhibitory exploit that we had not found by hand.


## Milestone 3 (20-21 September 2026): three behaviours, two sexes, choke points, search, the game

All numbers: 3 seeds per check; batch rows are mean +/- sd over 3 independent seed-sets; every behaviour passed the
full-brain fidelity check unless stated. Scripts: `run_redteam.py`, `batch.py`, `fidelity.py`, `sensitivity.py`,
`chokepoints.py`, `crosssex.py`, `game.py`, `calibrate.py` (all take `--sex female|male`).

### The fragility atlas

| behaviour | sex | healthy | BLOCK (smallest) | FORCE | disjoint cut-attacks | REPAIR | recovery |
|---|---|---|---|---|---|---|---|
| eat | female | 99 +/- 6 Hz | **2** (drive 2 GABA inputs of MN9) | **1** (DNge062) | 1 | 3 | 52 +/- 9 % |
| eat | male | 144 +/- 1 Hz | **4.7 +/- 0.5** cuts | **1** (DNge062) | 1 | 4.7 +/- 3.9 (1, 3, 10) | 36 +/- 5 % |
| escape | female | 234 +/- 1 Hz | **11.3 +/- 0.5** (drive giant-fiber inhibitors; **no cut-attack exists** within 10 lesions) | **1** (DNp70) | 0 | 3.7 +/- 0.5 | 43 +/- 3 % |
| escape | male | 410 Hz | **none found** (no strategy, no search, within the rules) | **2.3 +/- 0.5** | 0 | -- | -- |
| refuse_bitter (goal CORRUPT: make the fly eat sugar+bitter) | female | 0 Hz with sugar+bitter (107 Hz sugar alone) | cut 3 (bitter brake: CB0219, CB0571, CB0016) or **1** (drive DNge062; found by search, not greedy) | -- | 2 (3, 6) | -- | -- |
| refuse_bitter | male | 20 +/- 3 Hz with sugar+bitter (157 sugar alone) | 1 cut -> 42 Hz on the sub-network, **36 Hz on the full brain: fails the 40 Hz bar, reported as marginal** | -- | 4 (1, 2, 3, 4) | -- | -- |

What the atlas says: feeding is fragile (2-5 neurons), escape is an order of magnitude more robust and, in the male,
unbreakable under our rules; both are hijackable by 1-2 neurons. The model reproduces bitter gating of feeding without
being told to (sugar 107 Hz, sugar+bitter 0 Hz in the female); the gate can be broken by cutting 3 brake interneurons
or overridden by driving the feeding command neuron.

Left out, with the numbers: `groom` (JO stimulation drives descending neurons, but no citable grooming readout is typed
in either dataset and the data-driven picks differ between sides); `courtship` (male; ppk23/25 pheromone GRNs drive
the wing motor neurons at ~290 Hz but not pC1/aSP10, 0.7 Hz -- suggestive of song, not claimed). See `calibrate.py`.

### Is greedy the minimum? (beam search, `Attacker.search`)
Search matched greedy on eat (2 / 1, both sexes), escape (11 / 1; male: nothing), and beat it once: female corrupt
3 -> 1. Greedy sizes are reported as upper bounds; where search found smaller, the smaller is the atlas value.

### Choke points (`chokepoints.py`)
All minimal attacks we could enumerate per behaviour (5 to 15 sets), neuron frequency across them, and Spearman
correlation with three topology scores on the reachable sub-network:

| behaviour | sex | minimal attacks | neurons in >= half of them | rho: synapses onto readout | rho: shortest-path betweenness |
|---|---|---|---|---|---|
| eat | female | 5 (sizes 2-4) | CB0553, CB0493 (excitatory command interneurons); CB0862, CB0806, CB0465 (GABA) | +0.35 | +0.04 |
| eat | male | 15 (sizes 4-7) | GNG180, GNG184, GNG586 (GABA) | +0.31 | +0.05 |
| escape | female | 3 (size 11) | 7 giant-fiber inhibitors (CB3707, CB0563, CB0010, SAD014/017, CL038) | +0.19 | -0.04 |
| refuse_bitter | female | 5 (sizes 3-6) | CB0480, CB0571, CB0219, CB0016 (the bitter brake) | +0.13 | +0.08 |

None of the choke points lies on a shortest stimulus->readout path (betweenness 0); direct synapse count onto the
readout predicts them weakly. Where a circuit breaks is a property of its dynamics, not of its shortest paths.
Figures: `neuroredteam/atlas/<sex>/chokepoints_<behaviour>.png`.

### Conserved across sexes (`crosssex.py`, female `cell_type` <-> male `flywireType`)
| behaviour | matched types used by attacker/defender | conserved (used in both sexes) |
|---|---|---|
| eat | 14 | **7**: CB0465, CB0493, CB0553, CB0806, CB0862, DNge062, DNge080 -- the same command interneurons are cut and the same descending neuron (DNge062) is the 1-neuron hijack in both sexes |
| refuse_bitter | 11 | 2: CB0016, CB0219 (the bitter brake) |
| escape | 13 | 1: DNp70 (the force neuron); the giant fiber's inhibitors differ by sex |
Files: `neuroredteam/atlas/crosssex_<behaviour>.csv` (the only files where both sexes appear together).

### Deeper red / blue team (`game.py`)
* Stealth: all 5 minimal feeding attacks in the female leave escape intact (zero collateral).
* Iterated game (attacker attacks the repaired brain, defender repairs again; cumulative budget 12 neurons each):
  - boost-only defender (`Defender.greedy`): attacker wins in round 2 with a single cut -- the repair concentrated
    feeding drive onto 3 boosted neurons, which became the next choke point. **Repairs create new choke points.**
  - stronger defender (`Defender.robust`: may also depress inhibitory synapses onto the readout, silence
    suppressors, spread boosts, and probes each repair against a 1-2 neuron follow-up): survives 3 rounds; every
    repair still fragile to 1 more cut; attacker spends 5, defender 12, **exchange ratio 2.4**, attacker wins on budget.
  Feeding is defensible only at ~2.4x the attacker's cost, because MN9 has many inhibitory inputs the attacker can
  drive directly while the defender must act upstream.

### Checks (all in `neuroredteam/atlas/<sex>/`)
fidelity_*.json (sub-network vs full brain, every behaviour), sensitivity_*.csv (thresholds 30/15 to 60/5:
attack sizes unchanged; repair grows with a stricter ON bar; escape 60/5 finds no repair), batch logs, run logs.

## What is new and what is not (we do not overclaim)

Not new, reproduced by us: simulating the connectome (Shiu et al. 2024); the male connectome itself (MaleCNS, Cell 2026); perturbing it by hand (Shiu 2024; bioRxiv Dec 2025 stress tests); control-theory driver neurons in the worm (Yan et al. 2017); using connectome wiring as a neural network (conn2res 2024, FlyGM 2026, Dhiman 2026).

New, as far as our literature search found: automated search for the minimal perturbation that breaks a behaviour in a whole-brain behavioural simulation; an automated repair agent with a no-forcing test; the attack -> repair loop; a fragility atlas across behaviours; the same attack/repair benchmark run on the female and the male connectome; the iterated attack -> repair game with cumulative budgets; choke-point enumeration compared with topology; cross-sex matching of attacked cell types.

## Key references (all DOI-verified; full list in the deck)

Dorkenwald et al. 2024 Nature 634:124 (FlyWire, female) · MaleCNS: Cell 2026, doi 10.1016/j.cell.2026.08.015, data male-cns.janelia.org (male) · Shiu et al. 2024 Nature 634:210 (laptop simulation) · Pospisil et al. 2024 Nature 634:201 (effectome) · Yan et al. 2017 Nature 550:519 (worm control) · bioRxiv 10.64898/2025.12.14.694122 (stress tests) · Suarez et al. 2024 Nat Commun 15:656 (conn2res) · Dhiman 2026 arXiv:2604.04033 · Jin et al. 2026 arXiv:2602.17997 (FlyGM)
