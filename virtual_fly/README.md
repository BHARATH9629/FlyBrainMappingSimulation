# virtual_fly: the connectome decides, a 3-D fly shows it

Split screen. Left: a NeuroMechFly body (FlyGym 2.x, MuJoCo physics), head camera, walks to a sugar drop. Right: the
whole connectome as a dot cloud (139k female / 212k male neurons, real soma positions); every spike of the simulation is
drawn and fades, sugar neurons cyan, MN9 green, attacked neurons red, repair neurons blue. The 300 ms brain run is
played in slow motion while the fly decides. On arrival the simulated connectome is run with
the sugar taste neurons stimulated and the feeding motor neuron MN9 is read. If MN9 crosses the behaviour threshold
(40 Hz) the proboscis extends; otherwise the fly stands there. Three brains, straight from the atlas:
healthy, attacked (smallest BLOCK attack), repaired (attack + smallest repair). Works for the female and the male fly.

```bash
pip install flygym                      # brings mujoco; Python 3.14 is fine
python virtual_fly/demo.py --sex female # -> virtual_fly/out/female_{healthy,attacked,repaired}.mp4 (+ _end.png)
python virtual_fly/demo.py --sex male --camera head
```

Files: `body.py` (the fly: legs actuated for walking + rostrum/haustellum pitch actuated for the proboscis, cameras,
sugar drop), `brainview.py` (the dot-cloud renderer; needs `Brain.positions()` and `Brain.run(..., spikes=True)`),
`demo.py` (brain -> decision + spikes -> split-screen video with HUD).

Positions: female from `data/female_flywire/flywire_annotations.tsv` (Schlegel et al. 2024, downloaded by get_data);
male from `somaLocation` in the MaleCNS annotations. Sensory neurons have no soma in the volume; they are placed at the
centroid of their targets.

What is and is not the connectome, say it on the slide:
* connectome: the eat / no-eat decision (MN9 rate with the atlas intervention applied);
* NOT connectome: walking gait (FlyGym's CPG + rule controller), steering to the drop (a distance rule standing in
  for smell). Nobody has a connectome-driven nerve cord yet; the male CNS includes the leg motor neurons, so that is a
  possible future step.

Speed: ~10 wall seconds per simulated second of body + ~0.1 s per brain frame; one 3.8 s clip takes about a minute.
