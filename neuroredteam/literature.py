"""Does the model reproduce published fly experiments?

Each row is a manipulation someone has actually done in a fly, a prediction from that paper, and what our simulation
gives (mean over seeds 1-3). Verdicts: MATCH / MISMATCH / (partial). Nothing here is fitted -- the model has no free
parameters beyond the ones Shiu et al. published, so these are genuine tests.
Usage: python literature.py --sex female
"""
import argparse, json, os
import numpy as np
from brain import Brain
from behaviours import SUGAR_IDS, MN9_ID

ap = argparse.ArgumentParser()
ap.add_argument("--sex", default="female", choices=["female", "male"]); ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
brain = Brain(args.sex); SEEDS = range(1, args.seeds + 1)
here = os.path.dirname(os.path.abspath(__file__)); out_dir = os.path.join(here, "atlas", args.sex)

idx = {int(i): k for k, i in enumerate(brain.ids)}
sugar = [idx[i] for i in SUGAR_IDS if i in idx]
mn9 = idx[MN9_ID]
bitter_L = list(brain.by_type(subclass="bitter", side="L")); bitter = list(brain.by_type(subclass="bitter"))
loom_L = list(brain.by_type(types=["LC4", "LPLC2"], side="L")); loom_R = list(brain.by_type(types=["LC4", "LPLC2"], side="R"))
gf_L = list(brain.by_type(types=["DNp01"], side="L")); gf_R = list(brain.by_type(types=["DNp01"], side="R"))
lc16 = list(brain.by_type(types=["LC16"])); mdn = list(brain.by_type(types=["MDN"]))
dna02_L = list(brain.by_type(types=["DNa02"], side="L")); dna02_R = list(brain.by_type(types=["DNa02"], side="R"))


def rate(stim, record, **kw):
    """mean Hz over the recorded set, averaged over seeds"""
    return float(np.mean([np.mean([brain.run(stim, record=record, seed=s, **kw)["rates"][i] for i in record]) for s in SEEDS]))


TESTS = []


# checks on the circuit the model was built from, and on the rates our own thresholds were calibrated from: these
# cannot really fail, so they are reported separately from the independent tests.
CONSISTENCY = {"sugar GRNs -> MN9 fires", "silencing sugar GRNs abolishes it", "bitter + sugar -> MN9 eliminated",
               "bitter alone -> no feeding", "looming -> giant fibre DNp01",
               "specificity: sugar does not drive DNp01", "specificity: looming does not drive MN9"}


def test(name, paper, prediction, measured, ok, note=""):
    TESTS.append(dict(test=name, paper=paper, prediction=prediction, measured=measured, verdict=ok, note=note,
                      kind="consistency" if name in CONSISTENCY else "independent"))
    print(f"  {name:38s} {measured:28s} {ok}", flush=True)


print(f"{args.sex}: {len(sugar)} sugar GRNs, {len(bitter)} bitter GRNs, {len(loom_L)} left looming cells, "
      f"{len(gf_L)+len(gf_R)} DNp01, {len(lc16)} LC16, {len(mdn)} MDN, {len(dna02_L)+len(dna02_R)} DNa02", flush=True)

# --- feeding -----------------------------------------------------------------
base = rate(sugar, [mn9]); none = rate([], [mn9])
test("sugar GRNs -> MN9 fires", "Shiu et al. 2024 Nature (model + optogenetics); Gordon & Scott 2009",
     "MN9 fires (proboscis extension)", f"{base:.0f} Hz (vs {none:.0f} Hz unstimulated)", "MATCH" if base > 40 and none < 10 else "MISMATCH")

cut = rate(sugar, [mn9], lesion=sugar)
test("silencing sugar GRNs abolishes it", "Gr5a silencing abolishes PER to sugar (Dahanukar 2007; Marella 2006)",
     "MN9 silent", f"{cut:.0f} Hz", "MATCH" if cut <= 10 else "MISMATCH")

mix = rate(sugar + bitter_L, [mn9])
test("bitter + sugar -> MN9 eliminated", "Shiu et al. 2024: strong bitter activation eliminates MN9 firing to sugar (confirmed optogenetically)",
     "MN9 silent despite sugar", f"{mix:.0f} Hz (sugar alone {base:.0f} Hz)", "MATCH" if mix <= 10 else "MISMATCH")

bit = rate(bitter_L, [mn9])
test("bitter alone -> no feeding", "bitter GRNs do not drive PER (Scott lab)", "MN9 silent", f"{bit:.0f} Hz", "MATCH" if bit <= 10 else "MISMATCH")

half = rate(sugar + bitter_L[:len(bitter_L)//4], [mn9])
test("weak bitter -> graded suppression", "Shiu et al. 2024: suppression is graded with bitter strength (Ir94e reduces but does not eliminate)",
     "MN9 between silent and full", f"{half:.0f} Hz (full bitter {mix:.0f}, none {base:.0f})",
     "MATCH" if mix <= half <= base else "MISMATCH", "our only graded-dose test; Ir94e GRNs are not typed in the female annotation, so that specific comparison cannot be run")

# --- escape ------------------------------------------------------------------
gfL = rate(loom_L, [gf_L[0]] if gf_L else []) if gf_L else 0.0
test("looming -> giant fibre DNp01", "von Reyn et al. 2017 Neuron; Ache et al. 2019 Curr Biol",
     "DNp01 fires strongly", f"{gfL:.0f} Hz", "MATCH" if gfL > 80 else "MISMATCH")

gfR_from_R = rate(loom_R, gf_R) if gf_R else 0.0
gfL_from_L = rate(loom_L, gf_L) if gf_L else 0.0
ratio = gfR_from_R / gfL_from_L if gfL_from_L else 0.0
test("either eye drives its giant fibre", "azimuthal invariance of the giant-fibre escape circuit (Card lab, 2023)",
     "left and right similar", f"L {gfL_from_L:.0f} Hz, R {gfR_from_R:.0f} Hz (ratio {ratio:.2f})",
     "MATCH" if 0.5 <= ratio <= 2.0 else "MISMATCH")

gf_sugar = rate(sugar, gf_L) if gf_L else 0.0
test("specificity: sugar does not drive DNp01", "escape and feeding are separate pathways",
     "DNp01 silent", f"{gf_sugar:.0f} Hz", "MATCH" if gf_sugar <= 10 else "MISMATCH")
mn9_loom = rate(loom_L, [mn9])
test("specificity: looming does not drive MN9", "escape and feeding are separate pathways",
     "MN9 silent", f"{mn9_loom:.0f} Hz", "MATCH" if mn9_loom <= 10 else "MISMATCH")

# --- retreat and steering ----------------------------------------------------
mdn_r = rate(lc16, mdn) if lc16 and mdn else 0.0
test("LC16 -> moonwalker neurons (MDN)", "Wu et al. 2016 eLife (LC16 activation drives backward walking via MDN)",
     "MDN fires", f"{mdn_r:.0f} Hz", "MATCH" if mdn_r > 20 else "MISMATCH")

dl = rate(loom_R, dna02_L); dr = rate(loom_R, dna02_R)
test("threat on the right -> left DNa02", "Rayshubskiy et al. 2020: DNa02 activity predicts ipsilateral turning, so turning away from a right-side threat = left DNa02",
     "left DNa02 > right DNa02", f"left {dl:.0f} Hz, right {dr:.0f} Hz", "MATCH" if dl > dr + 5 else "MISMATCH")


# --- a published result the model does NOT reproduce (kept in the table on purpose) --------------------------------
TESTS.append(dict(test="odour direction -> asymmetric steering", kind="independent",
                  paper="flies track odour plumes by comparing the two antennae (Gaudry et al. 2013; Alvarez-Salvado et al. 2018)",
                  prediction="one-sided odour gives a left/right difference in DNa02",
                  measured="odour reaches the descending neurons, but DNa02 shows no graded left/right difference (calibrate_smell.log)",
                  verdict="MISMATCH",
                  note="expected: our stimulus model drives receptor neurons directly, with no antenna geometry and no "
                       "concentration gradient, so the asymmetry this test needs is absent by construction. A limit of "
                       "the setup, not evidence against the connectome."))

ind = [t for t in TESTS if t["kind"] == "independent"]
con = [t for t in TESTS if t["kind"] == "consistency"]
ok_of = lambda rows: sum(t["verdict"] == "MATCH" for t in rows)
print(f"\nindependent tests (not used to build or calibrate the model): {ok_of(ind)}/{len(ind)} reproduced")
print(f"consistency checks (the circuit the model was built on):       {ok_of(con)}/{len(con)}")

json.dump({"sex": args.sex, "seeds": args.seeds, "tests": TESTS}, open(os.path.join(out_dir, "literature.json"), "w"), indent=1)
with open(os.path.join(out_dir, "literature.md"), "w") as f:
    f.write(f"# Published experiments vs the model ({args.sex}, mean of seeds 1-{args.seeds})\n\n")
    f.write("Nothing here was fitted: the LIF parameters are the ones Shiu et al. published, and no threshold was\n"
            "changed to make a row pass. The table separates tests the model could genuinely fail from checks on the\n"
            "circuit it was built from.\n")
    sections = (("Independent tests", ind),
                ("Consistency checks -- the model was built on this circuit and our thresholds were set from these "
                 "rates, so they cannot really fail", con))
    for title, rows in sections:
        f.write(f"\n## {title}: {ok_of(rows)}/{len(rows)} reproduced\n\n")
        f.write("| experiment | published result | model | verdict |\n|---|---|---|---|\n")
        for t in rows:
            f.write(f"| {t['test']} | {t['paper']} | {t['measured']} | **{t['verdict']}** |\n")
    f.write("\n### Notes\n")
    for t in TESTS:
        if t.get("note"):
            f.write(f"\n- **{t['test']}**: {t['note']}\n")
print(f"saved atlas/{args.sex}/literature.md and .json")
