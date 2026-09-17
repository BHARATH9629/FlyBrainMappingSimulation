"""Rewrite the content of the existing deck around the two verified-novel contributions,
in simple language. Edits C:/Users/hp/Desktop/ConnectomeLab/EL_Phase1_Connectome_Lab.pptx IN PLACE."""
import sys, os
sys.argv = ["build_ppt2.py"]
src = open("build_ppt2.py", encoding="utf8").read()

def rep(old, new):
    global src
    assert old in src, "NOT FOUND: " + old[:70]
    src = src.replace(old, new)

# ---------- output goes to the existing deck ----------
rep('SRC, OUT = "template3.pptx", "EL_Phase1_Connectome_Lab.pptx"',
    'SRC, OUT = "template3.pptx", r"C:\\Users\\hp\\Desktop\\ConnectomeLab\\EL_Phase1_Connectome_Lab.pptx"')

# ---------- Slide 4: Introduction ----------
s4_start = src.index("# ===================== 4. Introduction")
s4_end = src.index("# ===================== 5. Literature Review")
src = src[:s4_start] + '''# ===================== 4. Introduction =====================
set_body(S[3], [
    (0, [("The big news: ", True),
         ("In October 2024 scientists finished mapping every neuron and every connection in a fruit fly's brain — "
          "139,255 neurons, 15 million connections — and released it free. In June 2026 the male brain and the "
          "brain-plus-nerve-cord maps followed. It is the first complete \\"wiring diagram\\" of an animal that can "
          "walk, fly, see and learn.", False)]),
    (0, [("What people can already do with it: ", True),
         ("run the whole brain on a laptop (Nature 2024); use its wiring as a neural network; type simple commands "
          "like \\"give the fly sugar\\" into a simulator. We have reproduced the laptop simulation ourselves: sugar "
          "on the tongue makes the feeding neuron fire; random input does not (slides 10–11).", False)]),
    (0, [("Two things nobody has done yet — our project: ", True),
         ("(1) an AI that runs its own experiments on the brain to discover which neurons cause a behaviour, plus an "
          "exam to score any such AI; (2) settling whether the fly's wiring is a better neural network than random "
          "wiring by testing it with realistic spiking neurons — the experiment missing from a 2026 debate.", False)]),
    (0, [("Why it matters: ", True),
         ("labs could test ideas on the simulated brain before touching a real animal; AI research learns whether "
          "copying brain wiring is worth it; and a fly brain does vision, flight and learning on about 1 milliwatt — "
          "understanding how is the road to low-power computing.", False)]),
], size=18)

''' + src[s4_end:]

# ---------- Slide 5: Literature Review ----------
s5_start = src.index("# ===================== 5. Literature Review")
s5_end = src.index("# ===================== 6. Problem Definition")
src = src[:s5_start] + '''# ===================== 5. Literature Review =====================
set_body(S[4], [
    (0, [("The map and the simulation:", True)]),
    (1, [("Dorkenwald et al. (2024, Nature) – ", True), ("the complete fly brain connectome (FlyWire), free to download.", False)]),
    (1, [("Shiu et al. (2024, Nature) – ", True), ("ran the whole map on a laptop with simple spiking neurons; it correctly predicted feeding, grooming and more. We reproduced this.", False)]),
    (1, [("Bates et al. (2026, Nature); MaleCNS (2026, Cell) – ", True), ("brain + nerve cord, and the male brain. Full FlyWire also runs on Intel's Loihi 2 brain-chip (2025).", False)]),
    (0, [("Brain wiring as a neural network — the open argument:", True)]),
    (1, [("Suárez et al. (2024, Nat. Commun., conn2res) – ", True), ("toolbox that uses real connectomes as reservoirs and compares them with fairly-scrambled versions. Real wiring wins for human, mouse, rat, monkey — but NOT for the fly.", False)]),
    (1, [("Jin et al. (Feb 2026, arXiv, FlyGM) – ", True), ("says the fly's exact wiring beats standard networks for controlling a simulated fly body.", False)]),
    (1, [("Dhiman (Apr 2026, arXiv) – ", True), ("says that advantage disappears when the comparison is fair (same starting point, degree-preserving scramble). Both sides used simplified, non-spiking neurons.", False)]),
    (0, [("AI agents that run experiments: ", True), ("exist for chemistry, materials and quantum physics (2025–26). For the fly brain only single-command tools exist (\\"give sugar\\" → one run); no agent plans its own experiments, and there is no benchmark to score one.", False)]),
    (0, [("Gap: ", True), ("(1) no benchmark or agent for automatic circuit discovery on a brain simulation; (2) nobody has tested whether the wiring advantage appears once neurons are realistic (spiking, inhibition respected).", False)]),
], size=14, sub_size=13)

''' + src[s5_end:]

# ---------- Slide 6: Problem Definition ----------
s6_start = src.index("# ===================== 6. Problem Definition")
s6_end = src.index("# ===================== 7. Objectives")
src = src[:s6_start] + '''# ===================== 6. Problem Definition =====================
set_body(S[5], [
    (0, [("Problem 1 — a map is not understanding. ", True),
         ("We have every wire in a brain, but finding which neurons cause a behaviour still means a scientist guessing: "
          "stimulate this, cut that, repeat for months. AI agents now run experiments by themselves in chemistry and "
          "physics, but nobody has built one for a brain — and nobody has an exam to check whether such an AI gets the "
          "right answer.", False)]),
    (0, [("Problem 2 — does nature's wiring actually help? ", True),
         ("Two 2026 papers disagree on whether the fly's real wiring makes a better neural network than random wiring. "
          "Both tested it with simplified neurons. Nobody has run the same fair test with realistic spiking neurons, "
          "where inhibitory connections actually inhibit. Our own first result (a piece of the fly brain playing "
          "Flappy Bird) found real wiring ≈ scrambled wiring with simplified neurons — so the question is live.", False)]),
    (0, [("Problem statement: ", True),
         ("Build a laptop platform that runs the complete fly connectome as a spiking neural network coupled to a "
          "virtual fly, and use it to (1) create CircuitBench — a benchmark that scores AI agents on discovering known "
          "behavioural circuits — together with the first such agent, and (2) test whether the computational advantage "
          "of evolved wiring over degree-preserving random wiring depends on the neuron model (rate-based vs spiking).", False)]),
], size=16)

''' + src[s6_end:]

# ---------- Slide 7: Objectives ----------
s7_start = src.index("# ===================== 7. Objectives")
s7_end = src.index("# ===================== 8. Methodology")
src = src[:s7_start] + '''# ===================== 7. Objectives =====================
set_body(S[6], [
    (0, [("Main objective: ", True),
         ("Run the real fly brain on a laptop and use it to answer two open questions: can an AI discover brain "
          "circuits by itself, and does evolution's wiring beat random wiring once neurons are realistic?", False)]),
    (0, [("Specific objectives:", True)]),
    (1, [("Brain engine + atlas: ", True), ("simulator of all 139,255 neurons with stimulate / record / cut, plus a name table (sugar, bitter, touch, vision, motor neurons); reproduce feeding, bitter-suppression and grooming from Nature 2024.", False)]),
    (1, [("CircuitBench (new): ", True), ("an exam of 4–6 circuits whose answers are known from the papers, with hidden answers, a fixed budget of experiments, and a scoring rule (right neurons found? how many experiments?).", False)]),
    (1, [("AI scientist (new): ", True), ("an LLM given the simulator as tools that plans, runs and interprets its own experiments; scored on CircuitBench against random search and a fixed strategy; compare two LLMs.", False)]),
    (1, [("Wiring test (new): ", True), ("same fly wiring, same degree-preserving scramble, same tasks — run once with simplified neurons and once with spiking neurons; report where the real wiring wins, ties or loses.", False)]),
    (1, [("Virtual fly + dashboard: ", True), ("a 2-D fly driven by the brain, and one web page where anyone can press \\"sugar\\", cut neurons, play the game, and chat with the agent.", False)]),
    (0, [("What is new: ", True), ("the benchmark and agent (nothing like it exists for a brain), and the spiking-vs-simplified wiring test (the missing experiment in the 2026 debate). The simulator, virtual fly and dashboard are our platform, not claimed as new.", False)]),
], size=16, sub_size=15)

''' + src[s7_end:]

# ---------- Slide 8: Methodology ----------
s8_start = src.index("# ===================== 8. Methodology")
s8_end = src.index("# ===================== 9. Methodology flowchart")
src = src[:s8_start] + '''# ===================== 8. Methodology =====================
set_body(S[7], [
    (0, [("Design: ", True), ("simulation experiments with proper controls. Every result is repeated over many random seeds and compared with published lab data or with a fair baseline.", False)]),
    (0, [("Data: ", True), ("public FlyWire v783 connectome (neurons, synapse counts, excitatory/inhibitory sign, cell-type labels). No data collection from people or animals.", False)]),
    (0, [("CircuitBench: ", True), ("ground truth = circuits published in Shiu et al. 2024 and follow-ups. The agent gets tools (stimulate, record, cut, look up wiring), a budget of N experiments, and must name the neurons. Score = overlap with the known circuit and experiments used. Baselines: random search, greedy \\"cut the strongest input\\", and a human-written script.", False)]),
    (0, [("Wiring test: ", True), ("take a sub-network (e.g. 3,000 neurons); build a degree-preserving scramble (each neuron keeps its number of connections, partners are shuffled); train only a read-out layer on 3–5 tasks (Flappy Bird done; time-series, spoken digits, image next). Run everything twice — simplified rate neurons and spiking neurons — and compare.", False)]),
    (0, [("Tools: ", True), ("Python, NumPy/SciPy, PyTorch, NetworkX, Claude/OpenAI API, pygame, FastAPI + web page. Laptop only; free Colab GPU if needed.", False)]),
    (0, [("Team: ", True), ("Brain engine + virtual fly (Govind) · Atlas + CircuitBench ground truth (Bharath) · AI scientist agent (Dhanush) · Wiring test + dashboard (Shreenidhi).", False)]),
], size=15)

''' + src[s8_end:]

# ---------- Slide 9: flowchart boxes ----------
rep('''row1 = [("1. Load the brain", "FlyWire data: 139,255 neurons,\\n15 M connections   ✔ done"),
        ("2. Build simulator", "spiking neurons in NumPy;\\nstimulate / record / cut   ✔ done"),
        ("3. Validate", "reproduce feeding (✔ done),\\nbitter, grooming vs Nature 2024"),
        ("4. Brain atlas", "cell-type labels + graph\\nqueries (paths, hubs)")]
row2 = [("5. Virtual fly", "2-D fly driven by motor\\nneurons: walk, escape, groom"),
        ("6. AI scientist", "LLM with tools plans and runs\\nits own experiments"),
        ("7. NN benchmark", "fly wiring vs random / MLP\\non learning tasks (PyTorch)"),
        ("8. Dashboard + paper", "web demo, report,\\nconference submission")]''',
'''row1 = [("1. Load the brain", "FlyWire data: 139,255 neurons,\\n15 M connections   ✔ done"),
        ("2. Build simulator", "spiking neurons in NumPy;\\nstimulate / record / cut   ✔ done"),
        ("3. Validate + atlas", "feeding ✔; bitter, grooming;\\nname table for neurons"),
        ("4. CircuitBench", "exam of known circuits,\\nhidden answers, scoring rule")]
row2 = [("5. AI scientist", "LLM plans + runs its own\\nexperiments; scored on the exam"),
        ("6. Wiring test", "fly vs fair scramble, simplified\\nvs spiking neurons, 3-5 tasks"),
        ("7. Virtual fly + dashboard", "2-D fly, game, chat —\\none web page"),
        ("8. Report + paper", "results, open code,\\nconference submission")]''')
rep('box(s, x, Y2, Wb, Hb, t, sub, fill=RED if i == 1 else NAVY)', 'box(s, x, Y2, Wb, Hb, t, sub, fill=RED if i in (0, 1) else NAVY)')
rep('box(s, x, Y1, Wb, Hb, t, sub)\n    if i < 3: arrow(s, x + Wb, Y1 + Hb / 2, x + Wb + GAP, Y1 + Hb / 2)',
    'box(s, x, Y1, Wb, Hb, t, sub, fill=RED if i == 3 else NAVY)\n    if i < 3: arrow(s, x + Wb, Y1 + Hb / 2, x + Wb + GAP, Y1 + Hb / 2)')
rep('''caption(s, 0.55, 5.65, 12.2, 0.9, "Steps 1–2 and the first validation are done (Sept 2026). Each later step is owned by one team "
        "member and plugs into the dashboard. Every experiment is repeated over 30 trials and compared with published results.", size=14)''',
    '''caption(s, 0.55, 5.65, 12.2, 0.9, "Red boxes = the new contributions. Steps 1–2 and the first validation are done (Sept 2026). "
        "Every result is repeated over many seeds and compared with published data or a fair baseline.", size=14)''')

# ---------- Slide 11: live demo text — mention Flappy ----------
rep('''    (0, [("December version: ", True), ("a 2-D fly walking in an arena, buttons for sugar / bitter / touch / light, a scalpel tool, and a chat box to the AI scientist.", False)]),''',
    '''    (0, [("Also working: ", True), ("3,000 of these neurons, unchanged, play Flappy Bird live (only a thin output layer is trained). With the pipe hidden, brain beats no-brain 5–10×; real wiring ≈ scrambled wiring with simplified neurons — the question our wiring test answers.", False)]),''')

# ---------- Slide 12: Proposed Outcomes ----------
s12_start = src.index("# ===================== 12. Proposed Outcomes")
s12_end = src.index("# ===================== 13. Timeline")
src = src[:s12_start] + '''# ===================== 12. Proposed Outcomes =====================
set_body(S[11], [
    (0, [("Expected results:", True)]),
    (1, "CircuitBench: an open exam (circuits, hidden answers, scoring script) that anyone can use to test an AI agent on a brain."),
    (1, "An AI scientist that passes it — e.g. given only \\"what makes the fly eat?\\", finds the same 3 command neurons we found by hand — with a table: accuracy and experiments-needed vs random search and a fixed strategy, for two different LLMs."),
    (1, "A wiring-test table: for each task, real wiring vs fair scramble, with simplified and with spiking neurons. Either outcome (advantage appears / does not) settles a piece of the 2026 debate."),
    (1, "A working platform: whole-brain simulator matching Nature 2024, a 2-D virtual fly, the Flappy Bird player, one dashboard."),
    (0, [("Impact: ", True), ("a yardstick for AI-driven neuroscience; a clear answer to \\"should AI copy brain wiring?\\"; labs can test ideas on the simulation before a real animal; the same tools transfer to the mouse and human maps now being built.", False)]),
    (0, [("Future work: ", True), ("more circuits in the exam; male-vs-female functional comparison (MaleCNS is 3 months old); running the spiking benchmark on a neuromorphic chip (Loihi / Raspberry Pi).", False)]),
], size=16, sub_size=15)

''' + src[s12_end:]

# ---------- Slide 13: Timeline ----------
s13_start = src.index("# ===================== 13. Timeline")
s13_end = src.index("# ===================== 14. Conclusion")
src = src[:s13_start] + '''# ===================== 13. Timeline =====================
set_body(S[12], [
    (0, [("Phase 1 – Foundation (Weeks 1–2, 17 Sep – 1 Oct)", True)]),
    (1, "Scope agreed with mentor; GitHub repo; simulator cleaned up (30-trial runs); cell-type name table loaded."),
    (0, [("Phase 2 – Validation + CircuitBench ground truth (Weeks 3–5, 1 Oct – 22 Oct)", True)]),
    (1, "Reproduce feeding, bitter-suppression, grooming; write down the known circuits as the exam; scoring script."),
    (0, [("Phase 3 – AI scientist + wiring test (Weeks 6–9, 22 Oct – 19 Nov)", True)]),
    (1, "Simulator exposed as LLM tools; agent loop with budget; run agent + baselines on CircuitBench."),
    (1, "Degree-preserving scramble; 3–5 tasks; run with simplified and with spiking neurons."),
    (0, [("Phase 4 – Virtual fly + dashboard (Weeks 8–11, 5 Nov – 3 Dec)", True)]),
    (1, "2-D fly arena driven by the brain; one web page with brain view, game, chat box, results tables."),
    (0, [("Phase 5 – Report + paper (Weeks 11–13, 3 Dec – 17 Dec)", True)]),
    (1, "Final demo, project report, open-source release, IEEE-format paper draft."),
], size=16, sub_size=15)

''' + src[s13_end:]

# ---------- Slide 14: Conclusion ----------
s14_start = src.index("# ===================== 14. Conclusion")
s14_end = src.index("# ===================== 15–16. References")
src = src[:s14_start] + '''# ===================== 14. Conclusion =====================
set_body(S[13], [
    (0, [("Summary: ", True), ("The complete fly brain runs on our laptop — we reproduced a Nature 2024 result and got a piece of the brain playing Flappy Bird. Simulating the brain is done; comparing its wiring to random wiring is done and currently disputed. We add the two missing pieces: an exam and an AI agent for discovering circuits automatically, and the spiking-neuron version of the wiring test.", False)]),
    (0, [("Final thought: ", True), ("This is where neuroscience and AI meet. A fly does vision, flight and learning with 139 thousand neurons and almost no power. The fastest way to learn what its wiring is good for is to run it, let an AI experiment on it, and test it fairly.", False)]),
    (0, [("Next steps: ", True), ("mentor approval → validation + exam ground truth (Oct) → agent and wiring test (Nov) → dashboard, report and paper (Dec).", False)]),
], size=18)

''' + src[s14_end:]

# ---------- References: add conn2res, Dhiman, Loihi 2 ----------
rep('''    "Jin, Z., Zhu, Y., Zhang, C., & Sui, Y. (2026). Whole-brain connectomic graph model enables whole-body locomotion control in fruit fly. arXiv preprint arXiv:2602.17997.",''',
    '''    "Jin, Z., Zhu, Y., Zhang, C., & Sui, Y. (2026). Whole-brain connectomic graph model enables whole-body locomotion control in fruit fly. arXiv preprint arXiv:2602.17997.",
    "Dhiman, N. (2026). Topological sensitivity in connectome-constrained neural networks. arXiv preprint arXiv:2604.04033.",
    "Suárez, L. E., Mihalik, A., Milisav, F., Marshall, K., Li, M., Vértes, P. E., Lajoie, G., & Misic, B. (2024). Connectome-based reservoir computing with the conn2res toolbox. Nature Communications, 15, 656. https://doi.org/10.1038/s41467-024-44900-4",
    "Neuromorphic simulation of Drosophila melanogaster brain connectome on Loihi 2. (2025). arXiv preprint arXiv:2508.16792.",''')
rep('set_body(S[15], [("none", r) for r in REFS_B], size=12, sub_size=12)', 'set_body(S[15], [("none", r) for r in REFS_B], size=11, sub_size=11)')

exec(compile(src, "build_ppt2_v3", "exec"))
