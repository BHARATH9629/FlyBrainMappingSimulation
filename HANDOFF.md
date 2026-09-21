# HANDOFF: how NeuroRedTeam works and what to do next

Read this before touching the code. Last updated 18 September 2026, after milestone 2 (robust agents + batch atlas) and the male-fly port.

## 0. The one-minute version

- Two flies, never mixed: `Brain("female")` = FlyWire v783 (139k neurons, 15 M connections, `data/female_flywire/`);
  `Brain("male")` = MaleCNS v1.0 (212k neurons incl. nerve cord, 26 M connections, `data/male_malecns/`). Every script takes
  `--sex female|male`; behaviours live in `BEHAVIOURS["female"]` / `BEHAVIOURS["male"]`; results in `atlas/female/` / `atlas/male/`.
- `neuroredteam/brain.py` runs it as a spiking network for 300 ms with optional interventions and returns firing rates.
- `neuroredteam/behaviours.py` says what "eat" means as a pass/fail test, and what the attacker and defender are allowed to do.
- `neuroredteam/agents.py` has the attacker and defender. Both are the same loop: choose an experiment, run the brain,
  check the behaviour, stop when the goal is met, then try to shrink the intervention set.
- `neuroredteam/run_redteam.py` runs attack then repair for one behaviour and writes an atlas entry + a full log.
- Milestone 2 result (18 Sep): every goal verified on 3 seeds, repeated over 3 seed-sets, zero variance. eat: block = 2
  neurons via the readout's own inhibitors (no cuts; beats the hand-found 3-cut), force = 1 neuron, 1 independent cut-attack,
  repair = 3 neurons, 52 % recovery. Runs are 1.2 s on a 13,131-neuron sub-network that reproduces the full brain exactly.

## 1. How the simulation works (brain.py)

Model: leaky integrate-and-fire, identical constants to Shiu et al. 2024 (`V0 -52 mV, threshold -45 mV, membrane 20 ms,
synapse 5 ms, refractory 2.2 ms, delay 1.8 ms, 0.275 mV per synapse`). Stimulated neurons get 150 Hz Poisson input.

`Brain.run(stimulate, t_ms=300, lesion=(), weaken=None, inject=None, bias=None, potentiate=None, record=())`

| Argument | Meaning | Used by |
|---|---|---|
| `stimulate` | neuron indices driven at 150 Hz (the world: sugar on the tongue) | behaviour |
| `lesion` | silence these neurons (all outgoing synapses removed) | attacker |
| `weaken` | `{idx: factor}` scale a neuron's outgoing synapses down | attacker |
| `inject` | `{idx: hz}` extra Poisson input (false sensory signal) | attacker |
| `bias` | `{idx: mV}` raise resting potential toward threshold, capped 0.5 mV below it (never fires on its own) | defender |
| `potentiate` | `{(pre, post): factor}` scale one synapse (plasticity) | defender |
| `record` | indices whose rate (Hz) is returned | behaviour |

Returns `{"rates": {idx: Hz}, "active": {idx: Hz for every neuron that fired}, "n_sub": size of sub-network}`.

Speed: it simulates only the sub-network reachable from the stimulus within `depth=3` hops over connections of
`>= 5` synapses, plus any neurons named in the interventions: 13,131 neurons for sugar, about 1.2 s per run. We checked
fidelity: depth 3 / 5 synapses gives the same MN9 rates as the full brain (107 Hz healthy, 0 Hz attacked); depth 2 does not
(73 Hz). The sub-matrix is cached; lesion/weaken are a diagonal scaling, potentiate a sparse add. When you add a behaviour,
re-check fidelity against `simulator/sugar_test.py`-style full runs before trusting the numbers.
Results are cached in memory by a hash of all arguments, so repeated experiments are free within a process.

Indices vs IDs (female): internally neurons are 0..138638 (row order of `Completeness_783.csv`). `brain.ids[i]` gives the
18-digit FlyWire ID, `brain.id2idx[id]` the index.
Male: indices are 0..211576 (sorted MaleCNS bodyId); `brain.ids[i]` is the bodyId, `brain.types` is a DataFrame with
`type`, `flywireType`, `superclass`, `class`, `side` per neuron, and `brain.by_type(["MN9"], side="L")` returns indices.
The male loader streams the 152 M-row weights file and keeps only edges between annotated neurons (32 s). Sign rule is
the same as Shiu et al.: `consensus_nt` in (gaba, glutamate) -> negative, everything else positive.
The male sub-network for sugar is 46k neurons (~2-6 s per run); `fidelity.py --sex male` confirmed it agrees with the whole
reachable CNS (185k neurons) on healthy / attacked / repaired. Logs and atlas entries store indices; convert with `brain.ids` for
papers or Codex look-ups (https://codex.flywire.ai/app/cell_details?root_id=<id>).

## 2. Behaviours and rules (behaviours.py)

A behaviour is: a stimulus set, a readout neuron, an ON threshold and an OFF threshold.
`Behaviour.is_normal(**interventions)` runs the brain twice (with and without stimulus) and returns
`(ok, on_hz, off_hz)`; ok means on >= ON and off <= OFF. This two-condition test is what stops the defender from
"repairing" by simply forcing the readout to fire.

Currently defined, female: `eat` (sugar GRN IDs and MN9 from the Shiu et al. example notebook; thresholds 40 / 10 Hz).
Male: `eat` (stimulus = types LB3b + LB3c left = sugar GRNs per the taste-connectome papers, readout = MN9 left) and
`eat_bitter` (stimulus LB1a-d = bitter GRNs; healthy MN9 should stay OFF, so this is the substrate for a 'corrupt' goal).
Other male type names worth knowing: LB3a water GRNs, PhG* pharyngeal, `superclass` in {descending_neuron, vnc_motor, ...}.

**TODO (Bharath):** on the male, `eat_bitter` already has IDs; write the `corrupt` goal (make MN9 fire to bitter) in agents.py.
On the female, add `refuse_bitter` (stimulus sugar + bitter, readout MN9, expected OFF), `groom` and `escape`
(stimulus and readout IDs from Shiu et al. 2024 supplementary tables and the FlyWire annotation CSV; the Dec 2025
bioRxiv names LC4/LPLC2 for looming and DNge031/CB0565 for feeding suppression). Put the IDs in `BEHAVIOURS` with a
source comment; run `is_normal()` on the healthy brain first and adjust thresholds until healthy = normal.

Rules: `RULES` (attacker) and `DEFENCE_RULES` (defender) are plain dicts: allowed moves, caps, and a simulation budget.
The agents read these; the readout neuron is always protected. **TODO (Shreenidhi):** enforce the caps inside
`Brain.run` too (raise on violation) so a future LLM agent cannot exceed them, and add a `motor` protected set once
motor neuron IDs are in the atlas.

## 3. Agents (agents.py)

Goals: `block` (readout <= OFF with stimulus) and `force` (readout >= ON without stimulus). Every check runs on
`CHECK_SEEDS = (1, 2, 3)` and must pass on all of them.
`Attacker.greedy()`: block = lesion the strongest active drivers one at a time; force = inject into the strongest
excitatory inputs. `Attacker.inject_inhibitors()`: block with no cuts by driving the readout's own inhibitory inputs
(this found the 2-neuron exploit). `Attacker.weaken_only()`: partial damage only. `Attacker.random()`: baseline.
`Attacker.best_of_all()`: smallest of the three. `redundancy()`: find an attack, exclude its neurons, search again
(how many independent attacks exist). All successful sets are minimised neuron by neuron. Cost = distinct neurons touched.

`Defender.greedy()`: with the attack fixed, rank neurons that still fire and still excite the readout; try bias only,
then potentiate only, then both, on the top 2/4/6/8/12; first valid repair wins; then drop neurons one at a time.

Every experiment goes through `Log.add(...)` and is printed and saved. The saved log is the "lab notebook" and is
what the dashboard will display.

**TODO (Dhanush):** LLM versions. Expose these functions as tools (JSON in/out): `find_neurons(label)`,
`inputs_to(idx)`, `outputs_of(idx)`, `run(...)`, `check(behaviour, result)`. Loop: system prompt with goal + rules +
budget, LLM picks a tool call, we execute, append result, repeat. Score exactly like greedy: intervention size and
simulations used. Start with the attacker; the defender is the same loop with the defence goal and rules.
Also add `Attacker.search()` (beam or genetic over lesion/weaken/inject sets seeded by greedy) as the middle baseline.

## 4. Running and outputs

Also: `python fidelity.py --sex male` (sub-network vs full brain), `python sensitivity.py --sex female` (thresholds).
`Defender.greedy()` now tries every plan and returns the smallest repair (it used to return the first that worked).

`python run_redteam.py --behaviour eat --sex female|male` writes `atlas/<sex>/<behaviour>.json`:

```
{"entry": {"behaviour", "attack": {"move", "neurons", "size", "simulations"},
           "repair": {"move", "neurons", "size", "on_hz", "off_hz", "simulations"},
           "healthy_on_hz", "healthy_off_hz", "total_seconds"},
 "log": [ {agent, step, hypothesis, intervention, result, goal_met, t}, ... ]}
```

**TODO (Shreenidhi):** `batch.py` that loops behaviours x goals x attackers x seeds (3 to 5) and writes one CSV; and
`atlas.py` that renders the table (mean +/- sd of attack size, repair size, recovery, simulations).

## 5. Dashboard and virtual fly

Virtual fly v1 exists: `virtual_fly/body.py` + `demo.py` (see `virtual_fly/README.md`). The connectome supplies only the
eat decision; FlyGym walks. Next steps (Govind): live mode (pygame/OpenCV window instead of mp4), head camera by
default so the proboscis is visible, and left/right descending-neuron rates -> turning (male has `descending_neuron`
superclass; female needs DN IDs).

FastAPI server exposing `run`, `attack`, `defend`, `atlas`; browser page with a brain view (dots by neuron), Attack /
Defend buttons streaming the log, and the atlas table. The 2-D virtual fly (Govind) reads motor/descending neuron
rates and moves a sprite; not needed for milestones 1 to 3.

## 6. Known caveats (say these out loud, do not hide them)

0. Female vs male differences (block 2 vs 4.7, repair 3 vs 4.7 +/- 3.9) come from different animals mapped by different
   labs. Done: male batch over 3 seed-sets, full-brain fidelity (`fidelity.py`), threshold sensitivity (`sensitivity.py`).
   Still to do: right-side and bilateral stimulus, and whether the attacked *cell types* match across sexes (`flywireType`).
   The male repair size is seed-sensitive (1, 3, 10) -- say so; do not quote a single number.

1. All results are about the model, not a living fly. They are predictions a lab could test.
2. Thresholds (40 / 10 Hz) are our choice. `sensitivity.py` shows attack sizes are unchanged from 30/15 to 60/5 Hz;
   repair size grows with a stricter ON threshold (female 3 -> 5, male 1 -> none within budget at 60/5).
3. One seed per experiment inside the agent loop; the atlas should average over seeds (batch.py).
4. The greedy attacker rediscovered a known answer. The interesting cases are behaviours where we do not know the
   answer, and non-obvious attacks (e.g. blocking feeding via the bitter pathway) that only search or an LLM finds.
5. The Flappy Bird side result showed the fly's exact wiring did not beat scrambled wiring as a generic reservoir.
   We report that honestly; it is why the project focuses on perturbation, not "is nature's wiring better".
6. Closest prior work: Shiu 2024 and the Dec 2025 bioRxiv stress tests perturbed this same model by hand for
   specific circuits; Yan 2017 found driver neurons in the worm with control theory. Cite them first.

## 7. Conventions

- Python 3.11+ (3.14 works). No GPU. Keep runs reproducible: pass `seed`.
- Do not commit anything under `data/*/` (100 MB female, 1.1 GB male); `python data/get_data.py [--male]` fetches them.
- Keep the sexes apart: new results go in `atlas/female/` or `atlas/male/`, never a mixed file; label every table with the sex.
- Commit small, descriptive messages. Put new experiments under `neuroredteam/` and results under `neuroredteam/atlas/`.
- When you add a behaviour, add one line to the README atlas table and one to `docs/` slide 10 if it changes the story.


## 9. Milestone 3 (20-21 Sep 2026): what was added and how to extend it

Scripts (all `--sex female|male`): `calibrate.py` (healthy rates for L/R/both, top descending/motor neurons: use it
to choose a readout and thresholds), `run_redteam.py` (now `--goal corrupt`, `--side`, beam search included, atlas
entry records greedy vs search), `fidelity.py`, `sensitivity.py`, `chokepoints.py`, `crosssex.py`, `game.py`.
`run_m3.sh <sex>` runs the whole pipeline for one sex into `atlas/<sex>/m3_*.log`.

Behaviour specs (`behaviours.py`): stimulus/readout can be IDs, a by_type query (`types`, `regex`, `subclass`,
`superclass`, `receptor`, `side`, `exclude_types`) or `{"union": [...]}`; `expected: "off"` marks a stimulus the fly
should refuse (goal `corrupt`); `sim` overrides depth/min_syn; `batch: False` keeps it out of batch/sensitivity;
`source` is the citation. Both sexes have a type table (`brain.types`, `brain.by_type`, `brain.type_of`); the female
one comes from `flywire_annotations.tsv` (Schlegel et al. 2024, fetched by `get_data.py`).

Adding a behaviour: write the spec with a `source`, run `calibrate.py` for L/R/both, set ON ~ 40 % of healthy and
OFF = max(10, 2x healthy OFF), then `run_redteam.py`, `fidelity.py`, `batch.py`, `sensitivity.py`, `chokepoints.py`,
`crosssex.py`. If the healthy brain does not pass, report the numbers and leave the row out (groom, courtship).

Agents: `Attacker(..., fixed=...)` attacks a brain that already carries interventions (used by the game and the
defender's probe); `Attacker.search()` = beam search over lesion/inject/weaken (budget `RULES["search_budget"]`,
one-seed scoring, full-seed verification); `Attacker.brakes()` = candidates for corrupt. `Defender.robust()` adds
moves `depress` (inhibitory synapse onto the readout x0.33), `silence` (up to 3 suppressors, never attacked neurons),
spread boosts, and an adversarial probe; it returns the smallest repair that survives a 1-2 neuron follow-up, or the
smallest repair flagged `fragile`. `merge_iv` stacks interventions.

Findings to carry into the paper: fragility spectrum (feeding 2-5, escape 11 / unbreakable); bitter gating reproduced;
choke points off the shortest paths; 7 conserved feeding types across sexes; repairs create new choke points;
feeding exchange ratio 2.4. Caveats: male corrupt is marginal at full brain (36 Hz vs 40 bar); male repair size is
seed-sensitive; escape attacks are inhibitor-injection only; greedy sizes are upper bounds.

Next (owners): literature validation table of attacked cell types vs published silencing results (Bharath);
2-3 more behaviours with citable readouts -- walking/turning via DNa02/MDN, water via LB3a, egg-laying via oviDN
(Bharath + Govind); pre-emptive defence and budget phase diagram (Shreenidhi); LLM attacker/defender on the same
tools (Dhanush); neurotransmitter-confidence resampling on the male (Govind).

## 8. Who owns what

| Person | Owns | First task |
|---|---|---|
| Govind | `brain.py` speed, virtual fly, dashboard plumbing | tighten sub-network to about 1 s per run |
| Bharath | `behaviours.py` IDs and thresholds, validation vs papers | add refuse_bitter, groom, escape |
| Dhanush | LLM attacker and defender, search baseline | tool wrappers + attacker loop |
| Shreenidhi | rules enforcement, batch runner, atlas table, dashboard UI | `batch.py` + `atlas.py` |
