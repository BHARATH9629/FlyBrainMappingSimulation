# Methods (DRAFT 1, 27 Sep 2026)

> Draft written from the code and the pre-registration log. Every number here should be checked against the cited
> file before submission. Sentences marked **[CAVEAT – needs team approval]** state a limitation and stay out of any
> shared version until approved.

## Connectomes

We used three synapse-resolution connectomes of adult *Drosophila melanogaster*:

| Data set | Animal | Neurons used | Source |
|---|---|---|---|
| FlyWire v783 | female, brain | 138,639 | Dorkenwald et al. 2024; Schlegel et al. 2024; model files of Shiu et al. 2024 |
| BANC v888 | female, brain + ventral nerve cord | all non-glial bodies | Bates et al. 2025 (BANC) [check exact citation] |
| MaleCNS v1.0 | male, brain + ventral nerve cord | 165,733 (status "Traced" or "Anchor") | Janelia/Google MaleCNS v1.0, male-cns.janelia.org (CC-BY) [check exact citation] |

For MaleCNS we kept only bodies with status *Traced* or *Anchor*; the full annotation table (211,577 bodies) also
contains orphan fragments (15,925), glia (11,864) and bodies marked unimportant (10,751). In all three data sets a
connection's weight is its synapse count, and neurons whose predicted transmitter is GABA or glutamate are
inhibitory (negative weight); all others excitatory. Cell types were matched across data sets by their FlyWire name
(FlyWire: its own type; BANC: the FAFB-matched type; MaleCNS: `flywireType`, else its own type). Side is the soma
side (MaleCNS: soma side, else root side).

## Spiking model

We used the leaky integrate-and-fire model of Shiu et al. (2024) unchanged: resting and reset potential −52 mV,
threshold −45 mV, membrane time constant 20 ms, synaptic time constant 5 ms, refractory period 2.2 ms, synaptic
delay 1.8 ms, weight per synapse 0.275 mV, time step 0.1 ms, 300 ms per simulation. Stimulated neurons receive
Poisson input at 150 Hz. Rates are spike counts over the 300 ms. Each simulation uses a fixed random seed, so every
result is exactly reproducible.

For speed, each simulation runs on the sub-network of neurons reachable from the stimulated (and injected) neurons
within 3 synaptic steps over connections of at least 5 synapses, plus the recorded and manipulated neurons. We
tested this approximation against the whole brain (all reachable neurons, any synapse count; see *Robustness*).

**Cross-data-set weight scale.** Synapse counts differ systematically between reconstructions. Over cell types
matched between data sets, FlyWire neurons receive a median 1.815× the input synapses of their BANC counterparts
(3,448 types; IQR 1.29–2.55) and 0.556× those of their MaleCNS counterparts (4,891 types; IQR 0.41–0.70). The
pre-registered primary analysis multiplies all BANC weights by 1.815 and all MaleCNS weights by 0.556; the raw counts
(× 1) were a pre-registered secondary analysis, and a sweep over scales tested the effect directly (*Results*).

## Pre-registration and safeguards

Hypotheses, inclusion rules, thresholds and analyses were written in `research/PREREGISTRATION.md` and committed to
version control before each confirmatory run; every later change is dated in that file's change log with its reason,
and was committed before the data it concerns were analysed. The attacker, defender, behaviour and simulator code
(`agents.py`, `behaviours.py`, `brain.py`) was frozen for the whole study (fingerprint `878888a2`); a guard script
checks 132 protected files before and after every research run and stops the run if any changed. When a new script
reused the existing machinery (parallel launcher, weight-scale option, model-variant option), we first re-ran
previously finished jobs through it and required byte-identical results.

## Systematic pathway screen

*Candidate inputs:* every cell type whose superclass is sensory or visual projection, present under the same name in
all three data sets, with 10–300 neurons on the left side (96 types). *Stage A:* for each input type and data set, the
left-side neurons were driven at 150 Hz (seed 1) and the descending neuron with the highest rate was recorded; every
(input type, output type + side) reaching ≥ 40 Hz in at least one data set became a candidate pathway (164).
*Stage B:* each candidate was tested in all three data sets on seeds 1–8 with the two-condition test (mean output
rate ≥ 40 Hz with the input, ≤ 10 Hz without). A pathway was **included** if it passed on 8/8 seeds in at least two of
the three data sets (128 of 164; 105 in all three).
**[CAVEAT – needs team approval]** In this model no neuron fires without input, so the "without input" half of the
test is always met; inclusion effectively rests on the "with input" condition.

*Unit of analysis.* Included pathways converge on 35 output groups (output cell type + side); 82 of the 128 end on
DNb05 (left or right). Because pathways sharing an output are not independent, one pathway per output group was
drawn at random (`numpy.random.default_rng(0)`, before any attack) and attacked in every data set where it passed:
87 data-set-runs ("fly-runs").

## Red team (attack)

The attacker's goal is to *block* a pathway: output ≤ 10 Hz with the input present. It may not touch the output
neuron(s) or motor neurons. Three strategies were run for every fly-run, and the smallest successful attack kept:

1. *Greedy lesion:* silence the output's active excitatory inputs, strongest first (rate × synapses), up to 10
   neurons, then remove any neuron not needed.
2. *Inject inhibitors:* drive the output's inhibitory inputs with extra 150 Hz Poisson input, up to 20 neurons.
3. *Weaken only:* scale down outgoing synapses of the strongest drivers (factor ≥ 0.5).

Each strategy has a budget of 40 simulations. Candidates are screened on seeds 1–3 and a result is accepted only if
it holds on seeds 1–8. **"No attack found" means no attack within these limits**, not that none exists.

## Blue team (repair)

After an attack, the defender (`Defender.restore`, target 0.8) searches for the smallest repair that brings the
output back to at least 80 % of its healthy rate (and at most 130 %) on every acceptance seed, without undoing the
attack. Repair moves: raise the resting potential of surviving drivers (≤ 6 mV, ≤ 12 neurons), strengthen specific
surviving synapses (≤ 3×, ≤ 12 synapses), and stop the output listening to hijacked inputs. A repair that makes the
output fire without the input ("cheating") is flagged.

## Verification

Every attack and every repair was re-tested on 15 seeds never used in the search (11–25) and graded per seed:
SOLID (15/15 met), MARGINAL (12–14), FAIL (< 12). All grades are reported; none were discarded.

## Hypotheses and tests

- **H1** (redundancy predicts cost). R1 = number of neurons with ≥ 10 excitatory synapses onto the output that fire
  in the healthy run; R2 = minimum vertex cut between input and output (excitatory edges ≥ 5 synapses, ≤ 4 steps;
  max-flow on a split-node graph). Spearman ρ between R1 and attack cost, and R2 and lesion-only cost; no attack found
  ranked above every found cost; ρ computed per data set and averaged, one-sided p from 10,000 within-data-set
  permutations. Confirmed if both ρ ≥ 0.6 with p < 0.05.
- **H2** (attacks depend on the output's direct inputs). The attack search was repeated with every direct presynaptic
  partner of the output forbidden (agents then fall back one step upstream). Confirmed if, in ≥ 75 % of fly-runs with a
  baseline attack, the forbidden-inputs attack costs ≥ 2× the baseline or is not found (goal met on ≥ 12/15 seeds).
- **H3** (repair costs more than attack). One-sided sign test on repair size − attack size (ties dropped); confirmed if
  repair is larger in ≥ 75 % of runs with p < 0.05.
- **H5** (weak spots are conserved across animals). *Overlap:* Jaccard index of the attack neurons' cell types between
  data sets for the same pathway; null = same number of types drawn uniformly from the types within two steps upstream
  of that data set's output (10,000 draws); statistic = mean Jaccard; confirmed if p < 0.05. An exploratory stricter
  null drew only from the output's direct inputs. *Transfer:* each attack was mapped by cell type onto the same pathway
  in the other data sets and tested on seeds 11–25 (breaks = goal met on ≥ 12/15); confirmed if ≥ 75 % break.
- Sensitivity analyses (exploratory): the same tests with each output group counted once.

## Scaling sweep

The screen attack set was re-run with BANC at × 1.0 and × 1.4 and MaleCNS at × 1.0. At each scale a pathway was first
re-checked with the 8-seed inclusion test and attacked only if it passed. Break rate = included pathways with an
attack found. Pre-registered comparison: Fisher's exact test, BANC × 1.0 vs FlyWire.

## Robustness controls (pre-registered)

- **A. Whole brain.** For all 65 attacked fly-runs, the healthy pathway, the attack and attack + repair were re-tested
  on seeds 11–25 using every reachable neuron (any synapse count). Holds if ≥ 90 % of SOLID/MARGINAL attacks and repairs
  still meet ≥ 12/15.
- **B. Model parameters.** Four variants (synaptic weight × 0.85 and × 1.15; threshold −46 and −44 mV) on 16 fly-runs
  drawn at random; the full attack search re-run under each after the inclusion check. Holds if Spearman ρ between
  baseline and variant attack cost is ≥ 0.7 for every variant (fewer than 8 included: not assessable).
- **C. Random attacks.** For each attack, 20 random interventions of the same size and move type drawn from the
  output's direct partners (≥ 5 synapses; inhibitory partners for injection), tested on seeds 11–25. Holds if the median
  share that breaks the pathway is ≤ 10 %.

## Software and data availability

Python 3.14 with NumPy, SciPy and pandas; all code, the pre-registration and every result file are in the project
repository (github.com/g0vindoh/ConnectomeLab, `research/`). Connectome data are public and downloaded by
`data/get_data.py`. AI assistance: code and analysis scripts were developed with an AI coding assistant (Claude,
Anthropic); the authors designed the study, made all decisions, checked the results and take full responsibility.
