# HANDOFF: how NeuroRedTeam works and what to do next

Read this before touching the code. Last updated 17 September 2026, after milestone 1.

## 0. The one-minute version

- The brain is a table of 15 million signed connections between 139,255 neurons (`data/`).
- `neuroredteam/brain.py` runs it as a spiking network for 300 ms with optional interventions and returns firing rates.
- `neuroredteam/behaviours.py` says what "eat" means as a pass/fail test, and what the attacker and defender are allowed to do.
- `neuroredteam/agents.py` has the attacker and defender. Both are the same loop: choose an experiment, run the brain,
  check the behaviour, stop when the goal is met, then try to shrink the intervention set.
- `neuroredteam/run_redteam.py` runs attack then repair for one behaviour and writes an atlas entry + a full log.
- Milestone 1 result: eat can be blocked by 3 neurons and repaired by 5. Verified across seeds. Same 3 neurons as the
  manual experiment. Everything else in the plan builds on these files.

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

Speed: it simulates only the sub-network reachable from the stimulus within `depth=5` hops over connections of
`>= 2` synapses, plus any neurons named in the interventions. For sugar that is still about 131k neurons, about 4 s per run.
**TODO (Govind):** tighten to `depth=4, min_syn=3` and check MN9 rates stay within a few Hz; target about 1 s per run.
Results are cached in memory by a hash of all arguments, so repeated experiments are free within a process.

Indices vs IDs: internally neurons are 0..138638 (row order of `Completeness_783.csv`). `brain.ids[i]` gives the
18-digit FlyWire ID, `brain.id2idx[id]` the index. Logs and atlas entries store indices; convert with `brain.ids` for
papers or Codex look-ups (https://codex.flywire.ai/app/cell_details?root_id=<id>).

## 2. Behaviours and rules (behaviours.py)

A behaviour is: a stimulus set, a readout neuron, an ON threshold and an OFF threshold.
`Behaviour.is_normal(**interventions)` runs the brain twice (with and without stimulus) and returns
`(ok, on_hz, off_hz)`; ok means on >= ON and off <= OFF. This two-condition test is what stops the defender from
"repairing" by simply forcing the readout to fire.

Currently defined: `eat` (sugar GRN IDs and MN9 from the Shiu et al. example notebook; thresholds 40 / 10 Hz).

**TODO (Bharath):** add `refuse_bitter` (stimulus sugar + bitter, readout MN9, expected OFF), `groom` and `escape`
(stimulus and readout IDs from Shiu et al. 2024 supplementary tables and the FlyWire annotation CSV; the Dec 2025
bioRxiv names LC4/LPLC2 for looming and DNge031/CB0565 for feeding suppression). Put the IDs in `BEHAVIOURS` with a
source comment; run `is_normal()` on the healthy brain first and adjust thresholds until healthy = normal.

Rules: `RULES` (attacker) and `DEFENCE_RULES` (defender) are plain dicts: allowed moves, caps, and a simulation budget.
The agents read these; the readout neuron is always protected. **TODO (Shreenidhi):** enforce the caps inside
`Brain.run` too (raise on violation) so a future LLM agent cannot exceed them, and add a `motor` protected set once
motor neuron IDs are in the atlas.

## 3. Agents (agents.py)

`Attacker.greedy()`: rank interneurons by (their firing rate) x (excitatory synapses onto the readout), lesion the top
1, 2, 3 ... until the readout is <= OFF threshold with the stimulus present, then try dropping each neuron to
minimise. `Attacker.random()`: same budget, random subsets of active neurons (the baseline to beat).

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

`python run_redteam.py --behaviour eat --attacker greedy|random` writes `atlas/<behaviour>_<attacker>.json`:

```
{"entry": {"behaviour", "attack": {"move", "neurons", "size", "simulations"},
           "repair": {"move", "neurons", "size", "on_hz", "off_hz", "simulations"},
           "healthy_on_hz", "healthy_off_hz", "total_seconds"},
 "log": [ {agent, step, hypothesis, intervention, result, goal_met, t}, ... ]}
```

**TODO (Shreenidhi):** `batch.py` that loops behaviours x goals x attackers x seeds (3 to 5) and writes one CSV; and
`atlas.py` that renders the table (mean +/- sd of attack size, repair size, recovery, simulations).

## 5. Dashboard and virtual fly (later)

FastAPI server exposing `run`, `attack`, `defend`, `atlas`; browser page with a brain view (dots by neuron), Attack /
Defend buttons streaming the log, and the atlas table. The 2-D virtual fly (Govind) reads motor/descending neuron
rates and moves a sprite; not needed for milestones 1 to 3.

## 6. Known caveats (say these out loud, do not hide them)

1. All results are about the model, not a living fly. They are predictions a lab could test.
2. Thresholds (40 / 10 Hz) are our choice. A different cut-off shifts repair size by a neuron or two.
3. One seed per experiment inside the agent loop; the atlas should average over seeds (batch.py).
4. The greedy attacker rediscovered a known answer. The interesting cases are behaviours where we do not know the
   answer, and non-obvious attacks (e.g. blocking feeding via the bitter pathway) that only search or an LLM finds.
5. The Flappy Bird side result showed the fly's exact wiring did not beat scrambled wiring as a generic reservoir.
   We report that honestly; it is why the project focuses on perturbation, not "is nature's wiring better".
6. Closest prior work: Shiu 2024 and the Dec 2025 bioRxiv stress tests perturbed this same model by hand for
   specific circuits; Yan 2017 found driver neurons in the worm with control theory. Cite them first.

## 7. Conventions

- Python 3.11+ (3.14 works). No GPU. Keep runs reproducible: pass `seed`.
- Do not commit `data/*.parquet` or `data/*.csv` (100 MB, GitHub limit); `python data/get_data.py` fetches them.
- Commit small, descriptive messages. Put new experiments under `neuroredteam/` and results under `neuroredteam/atlas/`.
- When you add a behaviour, add one line to the README atlas table and one to `docs/` slide 10 if it changes the story.

## 8. Who owns what

| Person | Owns | First task |
|---|---|---|
| Govind | `brain.py` speed, virtual fly, dashboard plumbing | tighten sub-network to about 1 s per run |
| Bharath | `behaviours.py` IDs and thresholds, validation vs papers | add refuse_bitter, groom, escape |
| Dhanush | LLM attacker and defender, search baseline | tool wrappers + attacker loop |
| Shreenidhi | rules enforcement, batch runner, atlas table, dashboard UI | `batch.py` + `atlas.py` |
