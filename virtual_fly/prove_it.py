"""PROVE IT: checks anyone can run live, to see that the demo is a real simulation of a real, public connectome.

  1. THE DATA      the wiring files: where they come from, their size and fingerprint, neurons and connections counted now
  2. LOOK IT UP    a neuron from the demo (e.g. MN9) with its FlyWire ID and strongest inputs -- open the public FlyWire
                   website and check the same neuron and the same partners
  3. YOU CHOOSE    pick a behaviour, pick any neurons to silence (a cell type or a FlyWire ID), pick a random seed:
                   the brain runs right now and answers
  4. SAME IN, SAME OUT   the same seed gives exactly the same number; different seeds (different random noise)
                   give slightly different numbers but the same behaviour
  5. PUBLISHED     three experiments done on real flies, re-run live on the model

Usage: python virtual_fly/prove_it.py          (menu)
       python virtual_fly/prove_it.py --auto   (every check once with example choices, no questions)
"""
import argparse, hashlib, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "neuroredteam"))
from brain import Brain
from behaviours import Behaviour, SUGAR_IDS, MN9_ID

ap = argparse.ArgumentParser(); ap.add_argument("--auto", action="store_true"); args = ap.parse_args()
DATA = os.path.join(ROOT, "data", "female_flywire")
SOURCES = {"Connectivity_783.parquet": "github.com/philshiu/Drosophila_brain_model (Shiu et al. 2024, Nature)",
           "Completeness_783.csv": "github.com/philshiu/Drosophila_brain_model (Shiu et al. 2024, Nature)",
           "flywire_annotations.tsv": "github.com/flyconnectome/flywire_annotations (Schlegel et al. 2024, Nature)"}
NAMES = {"eat": "eating (sugar -> MN9)", "drink": "drinking (water -> MN9)", "refuse_bitter": "refusing bitter (sugar + bitter -> MN9 stays quiet)",
         "escape": "escaping (looming -> giant fibre DNp01)", "retreat": "backing away (LC16 -> MDN)", "turn": "turning away (right-eye threat -> DNa02 left)"}


def line(c="-"): print(c * 78)


def ask(prompt, default):
    if args.auto: print(f"{prompt} {default}"); return str(default)
    a = input(f"{prompt} [{default}] ").strip()
    return a or str(default)


def fid(i): return int(brain.ids[i])


def verdict(b, hz):
    if b.expected == "off": return "fly REFUSES (MN9 quiet)" if hz <= b.off_thr else "fly SWALLOWS the bitter food" if hz >= b.on_thr else "partly"
    return "behaviour ON" if hz >= b.on_thr else "behaviour OFF"


def data_check():
    line("="); print("1. THE DATA: the real files the brain is built from"); line("=")
    for f, src in SOURCES.items():
        p = os.path.join(DATA, f); h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
        print(f"{f:28s} {os.path.getsize(p) / 1e6:7.1f} MB   sha256 {h.hexdigest()[:20]}...")
        print(f"{'':28s} downloaded from {src}")
    print(f"\ncounted right now: {brain.N:,} neurons, {brain.W.nnz:,} neuron-to-neuron connections,"
          f" {int(abs(brain.W).sum()):,} synapses")
    print("Anyone can download the same files from those public pages; the same sha256 = the very same data.")


def lookup():
    line("="); print("2. LOOK IT UP: check a neuron from the demo on the public FlyWire website"); line("=")
    i = brain.id2idx[MN9_ID]
    print(f"MN9, the feeding motor neuron the demo watches:\n  FlyWire ID {MN9_ID}   cell type {brain.type_of(i)}   "
          f"class {brain.types['class'].iloc[i]} / {brain.types['subclass'].iloc[i]}")
    print(f"  open:  https://codex.flywire.ai/app/cell_details?cell_id={MN9_ID}")
    print("  its strongest inputs in our data (synapse counts; + excites, - inhibits):")
    for j, w in brain.inputs_to(i, top=6):
        print(f"    {fid(j)}   {brain.type_of(j) or '(no type)':12s} {w:+5d} synapses")
    print("On the website the same partners appear at the top of MN9's inputs (the website may apply its own synapse")
    print("filter, so a count can differ by a few).")


def you_choose():
    line("="); print("3. YOU CHOOSE: your experiment, run right now"); line("=")
    keys = list(NAMES)
    for k, n in enumerate(keys, 1): print(f"  {k}. {NAMES[n]}")
    name = keys[int(ask("behaviour number?", 1)) - 1]; b = Behaviour(brain, name)
    print("Neurons to silence: a cell type (e.g. LB3, DNp01, LC4, or CB0701 = MN9), FlyWire IDs separated by commas, or 'none'.")
    what = ask("silence?", "none" if not args.auto else "LB3")
    seed = int(ask("random seed (any whole number)?", np.random.randint(1, 10_000) if not args.auto else 4242))
    lesion = []
    if what.lower() != "none":
        for tok in [t.strip() for t in what.split(",") if t.strip()]:
            if tok.isdigit():
                if int(tok) in brain.id2idx: lesion.append(brain.id2idx[int(tok)])
                else: print(f"  {tok}: not a FlyWire ID in this brain")
            else:
                found = brain.by_type(types=[tok]).tolist()
                print(f"  {tok}: {len(found)} neurons of that type" if found else f"  {tok}: no neurons of that type")
                lesion += found
    t0 = time.time(); h, _ = b.measure(True, seed=seed); t1 = time.time()
    m, r = b.measure(True, seed=seed, lesion=lesion) if lesion else (h, None)
    rd = ", ".join(f"{brain.type_of(i) or '?'} ({fid(i)})" for i in b.readouts)
    print(f"\n{NAMES[name]}; readout {rd}; pass bar {b.on_thr:.0f} Hz; seed {seed}")
    print(f"  healthy brain        : {h:6.1f} Hz  -> {verdict(b, h)}      ({t1 - t0:.1f} s to simulate)")
    if lesion: print(f"  {len(lesion):,} neurons silenced : {m:6.1f} Hz  -> {verdict(b, m)}")


def repeat():
    line("="); print("4. SAME IN, SAME OUT: a simulation, not a recording"); line("=")
    b = Behaviour(brain, "eat")
    a1 = b.measure(True, seed=7)[0]; brain._run_cache.clear(); a2 = b.measure(True, seed=7)[0]
    print(f"eating, seed 7, run twice from scratch: {a1:.3f} Hz and {a2:.3f} Hz  -> {'identical' if a1 == a2 else 'DIFFERENT'}")
    vals = [b.measure(True, seed=s)[0] for s in (11, 12, 13, 14, 15)]
    print("seeds 11-15 (different random noise in the neurons): " + ", ".join(f"{v:.1f}" for v in vals) + " Hz")
    print(f"  -> the numbers move a little, the behaviour does not: all {'above' if min(vals) >= b.on_thr else 'NOT all above'} the {b.on_thr:.0f} Hz bar")


def published():
    line("="); print("5. PUBLISHED: experiments done on real flies, re-run on the model now"); line("=")
    b = Behaviour(brain, "eat"); sugar = brain.idx(SUGAR_IDS); mn9 = brain.id2idx[MN9_ID]
    bitter = brain.by_type(subclass="bitter", side="L").tolist()
    s = b.measure(True, seed=1)[0]
    print(f"sugar taste neurons on         -> MN9 {s:5.1f} Hz   real flies: extend the proboscis to feed (Shiu et al. 2024; Gordon & Scott 2009)")
    x = b.measure(True, seed=1, lesion=list(sugar))[0]
    print(f"...with the sugar neurons silenced -> MN9 {x:5.1f} Hz   real flies: no feeding response (Dahanukar 2007; Marella 2006)")
    r = brain.run(list(sugar) + bitter, record=[mn9], seed=1, **b.sim)["rates"][mn9]
    print(f"sugar + bitter together         -> MN9 {r:5.1f} Hz   real flies: bitter stops feeding (Shiu et al. 2024, optogenetics)")


print("loading the female fly connectome (FlyWire v783) ..."); t = time.time(); brain = Brain("female")
print(f"loaded in {time.time() - t:.0f} s: {brain.N:,} neurons")
CHECKS = [("the data", data_check), ("look it up on the FlyWire website", lookup), ("you choose the experiment", you_choose),
          ("same in, same out", repeat), ("published experiments", published)]
if args.auto:
    for _, f in CHECKS: f(); print()
else:
    while True:
        line("="); print("PROVE IT"); [print(f"  {k}. {n}") for k, (n, _) in enumerate(CHECKS, 1)]; print("  q. quit")
        c = input("choose: ").strip().lower()
        if c == "q": break
        if c.isdigit() and 1 <= int(c) <= len(CHECKS): CHECKS[int(c) - 1][1](); input("\n(press Enter)")
