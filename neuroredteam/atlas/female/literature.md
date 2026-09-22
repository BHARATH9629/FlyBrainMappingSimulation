# Published experiments vs the model (female, mean of seeds 1-3)

Nothing here was fitted: the LIF parameters are the ones Shiu et al. published, and no threshold was
changed to make a row pass. The table separates tests the model could genuinely fail from checks on the
circuit it was built from.

## Independent tests: 4/5 reproduced

| experiment | published result | model | verdict |
|---|---|---|---|
| weak bitter -> graded suppression | Shiu et al. 2024: suppression is graded with bitter strength (Ir94e reduces but does not eliminate) | 71 Hz (full bitter 0, none 107) | **MATCH** |
| either eye drives its giant fibre | azimuthal invariance of the giant-fibre escape circuit (Card lab, 2023) | L 233 Hz, R 269 Hz (ratio 1.15) | **MATCH** |
| LC16 -> moonwalker neurons (MDN) | Wu et al. 2016 eLife (LC16 activation drives backward walking via MDN) | 42 Hz | **MATCH** |
| threat on the right -> left DNa02 | Rayshubskiy et al. 2020: DNa02 activity predicts ipsilateral turning, so turning away from a right-side threat = left DNa02 | left 62 Hz, right 0 Hz | **MATCH** |
| odour direction -> asymmetric steering | flies track odour plumes by comparing the two antennae (Gaudry et al. 2013; Alvarez-Salvado et al. 2018) | odour reaches the descending neurons, but DNa02 shows no graded left/right difference (calibrate_smell.log) | **MISMATCH** |

## Consistency checks -- the model was built on this circuit and our thresholds were set from these rates, so they cannot really fail: 7/7 reproduced

| experiment | published result | model | verdict |
|---|---|---|---|
| sugar GRNs -> MN9 fires | Shiu et al. 2024 Nature (model + optogenetics); Gordon & Scott 2009 | 107 Hz (vs 0 Hz unstimulated) | **MATCH** |
| silencing sugar GRNs abolishes it | Gr5a silencing abolishes PER to sugar (Dahanukar 2007; Marella 2006) | 0 Hz | **MATCH** |
| bitter + sugar -> MN9 eliminated | Shiu et al. 2024: strong bitter activation eliminates MN9 firing to sugar (confirmed optogenetically) | 0 Hz (sugar alone 107 Hz) | **MATCH** |
| bitter alone -> no feeding | bitter GRNs do not drive PER (Scott lab) | 0 Hz | **MATCH** |
| looming -> giant fibre DNp01 | von Reyn et al. 2017 Neuron; Ache et al. 2019 Curr Biol | 233 Hz | **MATCH** |
| specificity: sugar does not drive DNp01 | escape and feeding are separate pathways | 0 Hz | **MATCH** |
| specificity: looming does not drive MN9 | escape and feeding are separate pathways | 0 Hz | **MATCH** |

### Notes

- **weak bitter -> graded suppression**: our only graded-dose test; Ir94e GRNs are not typed in the female annotation, so that specific comparison cannot be run

- **odour direction -> asymmetric steering**: expected: our stimulus model drives receptor neurons directly, with no antenna geometry and no concentration gradient, so the asymmetry this test needs is absent by construction. A limit of the setup, not evidence against the connectome.
