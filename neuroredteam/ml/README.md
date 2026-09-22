# Small AI model trained on the simulator (female, eat)

Dataset: `ml/gen_dataset.py` -- 5,001 interventions on a pool of 417 neurons (357 active with sugar + the 80 strongest
MN9 inputs): every single lesion / inject / weaken (1,251), 3,750 random mixed sets of 2-8 moves. Each sample = MN9 rate
with and without sugar on the real LIF simulator (5.4 h; ~3.7 s per sample). Saved as `atlas/female/ml/dataset_eat.npz`.

## Surrogate brain (`train_surrogate.py`) -- predict MN9 from the intervention, no simulation
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

Note: an attempt to predict a neuron's single-cut importance from its wiring statistics alone (degree, strength, synapses
to MN9, cell class) did not work (Spearman 0.26); which neurons matter has to be measured by simulation, which is what
the attacker does.
