# Small AI models trained on the simulator (female, eat)

Dataset: `ml/gen_dataset.py` -- 5,001 interventions on a pool of 417 neurons (357 active with sugar + the 80 strongest
MN9 inputs): every single lesion / inject / weaken (1,251), 3,750 random mixed sets of 2-8 moves. Each sample = MN9 rate
with and without sugar on the real LIF simulator (5.4 h; ~3.7 s per sample). Saved as `atlas/female/ml/dataset_eat.npz`.

## 1. Surrogate brain (`train_surrogate.py`) -- predict MN9 from the intervention, no simulation
Held-out 20 % (1,001 samples):

| target | model | MAE | R^2 | decision | accuracy | precision | recall |
|---|---|---|---|---|---|---|---|
| MN9 with sugar | ridge (linear) | 9.7 Hz | 0.67 | "blocked" (<= 10 Hz) | 99.4 % | 0.75 | 0.38 (8 positives) |
| MN9 with sugar | gradient boosting | 10.3 Hz | 0.61 | | | | |
| MN9 without sugar | ridge | 3.9 Hz | 0.84 | "forced" (>= 40 Hz) | 98.5 % | 0.92 | 0.75 (48 positives) |
| MN9 without sugar | gradient boosting | 2.7 Hz | 0.85 | | | | |

Speed: ~1,000,000 predictions / s vs ~0.3 / s for the simulator. Finding: a *linear* model is as good as trees, i.e.
intervention effects on this circuit are close to additive. Sanity check: cutting the surrogate's four highest-weight
neurons (59455, 4477, 45977, 106730) blocks eating on the real simulator (0 / 0 / 7 Hz on seeds 1-3; predicted -10 Hz);
the greedy attacker's 3-cut block shares neuron 59455. Blocks are rare in random sets (1 %), so recall for "blocked" is
low: the surrogate is a fast pre-filter, every candidate is still verified on the simulator.

## 2. Choke-point predictor (`train_chokepoints.py`) -- can wiring alone tell which neurons matter?
Labels = measured MN9 drop when each of the 417 neurons is cut / driven alone. 62 neurons drop MN9 by >= 20 Hz when cut;
none blocks alone (consistent with block size >= 2). Gradient boosting, 5-fold CV, features = degree, strength, synapses
to MN9, 2-hop synapses to MN9's drivers / inhibitors, firing rate, cell class:

| label | best single feature | model Spearman | top-10 overlap |
|---|---|---|---|
| drop when cut | firing rate +0.36 | +0.26 (MAE 7.9 Hz) | 0 / 10 |
| drop when driven | is-inhibitory +0.32 | +0.32 (MAE 14.3 Hz) | 7 / 10 |
| appears in a minimal attack (8 of 417) | -- | AUC 0.94, average precision 0.24 (chance 0.02) | |

Reading: which neurons hurt when *driven* is partly predictable (they are inhibitory), but which neurons hurt when *cut*
is not readable from local wiring statistics (rho 0.26, none of the true top-10 recovered). This is the quantitative
version of the earlier finding that choke points lie off the shortest paths: you have to simulate to find them.
Per-neuron table: `atlas/female/ml/neuron_effects_eat.csv`.
