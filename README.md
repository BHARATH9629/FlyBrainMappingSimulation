# NeuroRedTeam

**Attack and repair testing of a real fly brain.**
RVCE Experiential Learning, Semester 3 (Sept to Dec 2026).
Team: Govind, Bharath, Dhanush A, Shreenidhi Chalmi.

## What this is, in one paragraph

Scientists mapped every neuron and connection in a fruit fly's brain (FlyWire, Nature 2024: 139,255 neurons,
15 million connections) and released it free. A follow-up paper showed the whole map can be simulated on a laptop
and behaves like a real fly. We run that simulation and treat the brain like a system under security testing:
an **attacker agent** searches for the smallest set of allowed changes that breaks a behaviour, a **defender agent**
searches for the smallest stimulation that restores it, and we record the results for every behaviour we can
measure. The output is a **fragility atlas** of fly behaviours and a benchmark for the agents.

## Status (17 September 2026)

| Piece | Status | Evidence |
|---|---|---|
| Whole-brain simulation, matches Nature 2024 | done | sugar -> feeding neuron MN9 fires (95 to 107 Hz); random input -> 0 Hz |
| Behaviour "eat" as a pass/fail test | done | on >= 40 Hz with sugar AND off <= 10 Hz without |
| Attack rules (attack surface) | done | `neuroredteam/behaviours.py` |
| Attacker, greedy version | **done, verified** | finds the 3-neuron block in 6 simulations; set is minimal; same neurons as found by hand; holds across seeds |
| Defender, greedy version | **done, verified** | 5-neuron repair (bias + potentiate); 40 to 50 Hz with sugar, 0 Hz without, across seeds |
| First fragility-atlas entry (eat) | done | attack size 3, repair size 5, recovery about 47 % |
| Attacker, random baseline | code written, not yet run | `Attacker.random()` |
| Goals "force" (eat without sugar) and "corrupt" (eat bitter) | not yet | needs bitter neuron IDs |
| Behaviours groom, escape, refuse-bitter | not yet | needs neuron IDs from papers / FlyWire annotations |
| LLM attacker and defender | not yet | same tools; needs an API key |
| Fragility atlas over all behaviours | 1 of 4 | batch runner is trivial once behaviours exist |
| Virtual fly + web dashboard | not yet | |
| Side result: 3,000 fly neurons play Flappy Bird | done | `flappy/`; real wiring did NOT beat scrambled wiring, we say so |

## Quick start

```bash
pip install numpy scipy pandas pyarrow matplotlib pygame-ce
python data/get_data.py            # downloads the 100 MB connectome once (not stored in the repo)

cd neuroredteam
python run_redteam.py              # attack + repair of 'eat', about 3 minutes, writes atlas/eat_greedy.json
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
| `neuroredteam/` | **the project.** `brain.py` fast simulator + interventions + wiring queries; `behaviours.py` behaviour tests and rules; `agents.py` attacker and defender; `run_redteam.py` one-command run; `atlas/` results and full experiment logs |
| `simulator/` | the original whole-brain reproduction (`sugar_test.py`), lesion tracing, animations, defender feasibility tests |
| `flappy/` | the Flappy Bird side result: `flappy_fly.py` (train + compare), `live_flappy.py` (live game), `tune.py` |
| `data/` | `get_data.py` downloads `Connectivity_783.parquet` and `Completeness_783.csv` (gitignored, 100 MB) |
| `results/`, `animations/` | charts, GIFs, the methodology flowchart PNG |
| `docs/` | `NeuroRedTeam_Phase1.pptx` (also at repo root), explainer PDF, and the scripts that generate them |
| `HANDOFF.md` | how the code works, who owns what, what to do next, known caveats |

## The fragility atlas so far

| Behaviour | Healthy | Attack (smallest block found) | Repair (smallest restore found) | Recovery |
|---|---|---|---|---|
| eat | MN9 107 Hz with sugar, 0 without | lesion 3 neurons: 720575940623211725, 720575940627099338, 720575940632252743 | bias +6 mV and x2 synapses on 5 surviving inputs | 50 Hz with sugar, 0 Hz without (47 %) |

## What is new and what is not (we do not overclaim)

Not new, reproduced by us: simulating the connectome (Shiu et al. 2024); perturbing it by hand (Shiu 2024; bioRxiv Dec 2025 stress tests); control-theory driver neurons in the worm (Yan et al. 2017); using connectome wiring as a neural network (conn2res 2024, FlyGM 2026, Dhiman 2026).

New, as far as our literature search found: automated search for the minimal perturbation that breaks a behaviour in a whole-brain behavioural simulation; an automated repair agent with a no-forcing test; the attack -> repair loop; a fragility atlas across behaviours.

## Key references (all DOI-verified; full list in the deck)

Dorkenwald et al. 2024 Nature 634:124 (FlyWire) · Shiu et al. 2024 Nature 634:210 (laptop simulation) · Pospisil et al. 2024 Nature 634:201 (effectome) · Yan et al. 2017 Nature 550:519 (worm control) · bioRxiv 10.64898/2025.12.14.694122 (stress tests) · Suarez et al. 2024 Nat Commun 15:656 (conn2res) · Dhiman 2026 arXiv:2604.04033 · Jin et al. 2026 arXiv:2602.17997 (FlyGM)
