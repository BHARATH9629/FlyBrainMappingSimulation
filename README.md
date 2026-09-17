# Connectome Lab

Running a real fruit-fly brain (FlyWire v783: 139,255 neurons, 15.1 M synapses) as a spiking
neural network on a laptop — with a virtual fly, an AI scientist, and a neural-network benchmark.

RVCE Experiential Learning, Semester 3 (Sept–Dec 2026).
Team: Govind · Bharath · Dhanush A · Shreenidhi Chalmi

## Folder layout

| Folder | What is in it |
|---|---|
| `data/` | The connectome: `Connectivity_783.parquet` (15 M signed connections) and `Completeness_783.csv` (neuron IDs). From github.com/philshiu/Drosophila_brain_model (Shiu et al., Nature 2024). |
| `simulator/` | `sugar_test.py` — the whole-brain leaky integrate-and-fire simulator + sugar-vs-random experiment. `lesion_test.py` — path tracing and neuron cutting. `animate.py`, `animate2.py` — the two animations. |
| `results/` | Result charts (sugar vs random; lesion experiment). |
| `animations/` | `brain_demo.gif` (signal spreading through the brain), `fly_feeding_demo.gif` (brain + cartoon fly + surgery at 350 ms). |
| `docs/` | Phase-1 PPT (RVCE template) and the simple-words explainer PDF. `generators/` holds the Python scripts that build them from the template. |

## Run it

```bash
pip install numpy scipy pandas pyarrow matplotlib
cd simulator
python sugar_test.py      # ~80 s: sugar -> MN9 fires (95-98 Hz); random -> 0 Hz; saves sugar_test.png
python lesion_test.py     # traces tongue->MN9 path (3 synapses) and cuts neurons
python animate2.py        # regenerates fly_feeding_demo.gif (~2 min)
```

Everything runs on a normal laptop (no GPU). One second of brain time ≈ 35 s.

## Results so far (17 Sep 2026)

- Sugar on the tongue → feeding motor neuron MN9 fires at 95–98 Hz in 3/3 runs; random input → 0 Hz in 3/3 runs.
  Active-neuron and spike counts match Shiu et al. (Nature 2024).
- Shortest strong-excitatory path from sugar neurons to MN9: 3 synapses.
- Cutting the 3 strongest command neurons (720575940623211725, 720575940627099338, 720575940632252743)
  drops MN9 from 91 Hz to 1 Hz; cutting 10 random neurons changes nothing.

## Key references

- Dorkenwald et al. (2024) Neuronal wiring diagram of an adult brain. Nature 634:124–138.
- Shiu et al. (2024) A Drosophila computational brain model reveals sensorimotor processing. Nature 634:210–219.
- Lappalainen et al. (2024) Connectome-constrained networks predict neural activity across the fly visual system. Nature 634:1132–1140.
- Bates et al. (2026) Distributed control circuits across a brain-and-cord connectome. Nature.
- Full list (16 refs, all DOI-verified) in `docs/EL_Phase1_Connectome_Lab.pptx`, slides 15–16.
