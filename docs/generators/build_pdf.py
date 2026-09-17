"""Simple-words explainer PDF for the Connectome Lab EL project."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table,
                                TableStyle, KeepTogether, ListFlowable, ListItem)

OUT = "Connectome_Lab_Project_Explained_Simply.pdf"
NAVY = colors.HexColor("#002060")
RED = colors.HexColor("#C00000")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EEF2F8")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=RED, fontSize=20, spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=14, spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("Body", parent=ss["Normal"], fontSize=11, leading=16, spaceAfter=7)
QUOTE = ParagraphStyle("Quote", parent=BODY, backColor=LIGHT, borderPadding=8, leftIndent=6, rightIndent=6,
                       textColor=NAVY, fontSize=12, leading=17, spaceBefore=16, spaceAfter=16)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=9.5, leading=13, textColor=GREY)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=26, textColor=NAVY, leading=32, spaceAfter=6)
SUB = ParagraphStyle("S", parent=BODY, fontSize=13, textColor=GREY, alignment=1, spaceAfter=18)

def P(t, st=BODY):
    return Paragraph(t, st)

def bullets(items, st=BODY):
    return ListFlowable([ListItem(P(i, st), leftIndent=12) for i in items], bulletType="bullet",
                        start="•", leftIndent=14, bulletFontSize=10, spaceAfter=6)

def table(rows, widths, header=True):
    cell = ParagraphStyle("c", parent=BODY, fontSize=10, leading=13, spaceAfter=0)
    data = [[P(c, cell) for c in r] for r in rows]
    if header:
        data[0] = [P(f"<font color='white'><b>{c}</b></font>", cell) for c in rows[0]]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t

story = []

# ---------------- Cover ----------------
story += [Spacer(1, 3 * cm),
          P("Connectome Lab", TITLE),
          P("Running a real fly brain as a neural network — and letting an AI agent discover how it works", SUB),
          P("<b>Experiential Learning, Semester 3 — the project explained in simple words</b>",
            ParagraphStyle("c", parent=BODY, alignment=1, textColor=NAVY)),
          P("Team: Govind, Bharath, Dhanush A, Shreenidhi Chalmi", ParagraphStyle("c2", parent=BODY, alignment=1)),
          Spacer(1, 1.2 * cm),
          P("<b>Read this before the mentor meeting.</b> It explains what the project is, what we have already "
            "done, what we will build, why it matters, and how to answer the questions a teacher is likely to ask. "
            "No biology background is needed.", QUOTE),
          PageBreak()]

# ---------------- 1. One-line summary ----------------
story += [P("1. The project in one line", H1),
          P("<b>Scientists mapped every wire in a fruit fly's brain and gave it away for free. We run that brain on a "
            "laptop as a neural network, make it produce real behaviour, test whether nature's wiring is a better AI "
            "design than ours, and build an AI agent that experiments on the brain by itself.</b>", QUOTE),
          P("If someone asks \"what is your project?\" — say that sentence. Everything else in this document just "
            "explains the pieces.")]

# ---------------- 2. Background ----------------
story += [P("2. The background story (what the scientists did)", H1),
          P("A brain is made of neurons (cells) connected by synapses (wires). A complete map of all the wires is "
            "called a <b>connectome</b>. Until recently we only had this for a tiny worm with 302 neurons."),
          P("In October 2024, after about 10 years of work, an international team (the <b>FlyWire</b> project, "
            "Princeton and others) published the complete connectome of an adult fruit fly:"),
          bullets(["<b>139,255 neurons</b> — every single one, traced under an electron microscope",
                   "<b>about 50 million synapses</b> (15 million connections between neuron pairs)",
                   "for each connection: how strong it is, and whether it excites or inhibits the next neuron",
                   "it is <b>free to download</b> — about 100 MB of tables"]),
          P("The fruit fly is the first animal with a complete brain map that can also walk, fly, see, smell, learn "
            "and remember. That is why it matters so much."),
          P("<b>Timeline of the maps (so nobody mixes them up):</b>"),
          bullets(["<b>Oct 2024</b> — FlyWire: complete <b>female brain</b> (Nature). This is the dataset we use.",
                   "<b>May 2025</b> — complete inventory of the fly <b>visual system</b> (Nature).",
                   "<b>June 2026</b> — BANC: female <b>brain + nerve cord</b> together (Nature); and MaleCNS: the complete "
                   "<b>male</b> nervous system, 166,700 neurons (Cell). These are the ones in the recent Instagram/news videos."]),
          P("In the same month, a second paper (Shiu et al., <i>Nature</i>) did something surprising: they took the "
            "map, treated each neuron as a very simple \"leaky bucket\" that fires when it fills up, connected them "
            "exactly as the map says — and ran the whole brain on a laptop. When they \"put sugar on the tongue\" of "
            "the simulated fly, the neuron that makes a real fly extend its mouth to eat started firing. They then "
            "checked in real flies, and the prediction was correct.")]

# ---------------- 3. Already done ----------------
story += [P("3. What we have already done (proof it works)", H1),
          P("We did not want to promise something we could not do, so we tested it first. On 16–17 September we "
            "downloaded the real data and wrote our own simulator (about 90 lines of Python). Results:"),
          table([["Experiment", "What happened"],
                 ["Give the fly <b>sugar</b> (stimulate 20 sugar-sensing neurons)",
                  "376–389 neurons became active; the <b>feeding motor neuron (MN9) fired at 95–98 Hz</b> in every one of 3 repeated runs — the fly \"wants to eat\""],
                 ["Give the fly <b>random</b> input (21 random neurons)",
                  "Only 57–141 neurons active; <b>MN9 fired 0 times in all runs</b> — no feeding"],
                 ["Trace the wiring from tongue to mouth muscle", "Only <b>3 synapses</b> — sensation to action in 3 hops"],
                 ["<b>Cut 3 specific neurons</b> (the \"eat\" command cells)", "Feeding dropped from 91 Hz to <b>1 Hz</b> — feeding stops"],
                 ["Cut 10 <b>random</b> neurons", "No change (91 Hz) — the brain is robust to random damage"]],
                [7 * cm, 9.5 * cm]),
          Spacer(1, 8),
          P("These numbers match the Nature paper (about 400 active neurons, MN9 firing on sugar). One second of "
            "brain activity takes 30–40 seconds on a normal laptop. No GPU, no cloud, no special hardware. "
            "We re-ran everything on 17 September with fresh random seeds and got the same result.")]
try:
    story += [Image("../fly/sugar_test.png", width=15 * cm, height=6 * cm),
              P("Figure 1: left — every neuron that fired for sugar vs random input; right — MN9 fires only for sugar.", SMALL),
              Image("../fly/lesion_test.png", width=12 * cm, height=6.2 * cm),
              P("Figure 2: \"digital brain surgery\" — cutting the right 3 neurons stops feeding; cutting 10 random ones does nothing.", SMALL)]
except Exception:
    pass

story += [PageBreak()]

# ---------------- 4. What we will build ----------------
story += [P("4. What we are going to build (5 parts)", H1),
          P("Think of it as a small laboratory in software. Five parts, each owned by one team member, all meeting "
            "in one web page."),
          P("Part 1 — Brain engine (Govind)", H2),
          P("The simulator itself, made faster and cleaner. Four commands: <b>stimulate</b> (turn some neurons on), "
            "<b>record</b> (see which neurons fired), <b>cut</b> (remove neurons), <b>run</b> (simulate for some time). "
            "Every experiment is repeated 30 times so results are reliable."),
          P("Part 2 — Brain atlas (Bharath)", H2),
          P("A labelled map. The raw data only gives ID numbers; the atlas adds names: \"this is a sugar sensor\", "
            "\"this is a leg motor neuron\", \"this is a visual neuron\". It also answers wiring questions: "
            "\"shortest path from the eye to the legs?\", \"which neurons feed into this one most strongly?\", "
            "\"which neurons are the hubs?\""),
          P("Part 3 — AI scientist (Dhanush)", H2),
          P("A large language model (like Claude or GPT) that is given the four engine commands and the atlas as "
            "<b>tools</b>. You type a question — \"what makes the fly eat?\" — and the AI plans experiments, runs them, "
            "reads the numbers, decides what to try next, and writes a short report with graphs. We check it by "
            "asking it to rediscover circuits that scientists already know. Nobody has done this on a connectome yet."),
          P("Part 4 — Neural-network benchmark (Shreenidhi)", H2),
          P("The brain's wiring is loaded into PyTorch as a neural network. We give it learning tasks (recognising "
            "spoken digits, predicting a signal, classifying small images) and compare it with random networks and "
            "standard networks of the same size. Key question: <b>is the wiring evolution produced better than "
            "wiring we make up?</b> We also slowly \"shuffle\" the fly's wiring and plot how performance drops — the "
            "\"how much of the fly is left\" curve."),
          P("Part 5 — Dashboard (Shreenidhi + everyone)", H2),
          P("A web page with a picture of the brain. Press <b>Sugar</b> and watch neurons light up. Click neurons to "
            "cut them. A chat box talks to the AI scientist. This is what we show in the demo.")]

# ---------------- 5. AI mapping ----------------
story += [P("5. How this connects to AI and neural networks", H1),
          P("The key idea: <b>the fly brain already IS a neural network</b> — just one designed by evolution instead "
            "of by engineers. Compare:"),
          table([["Normal neural network (e.g. ChatGPT, image AI)", "Fly brain"],
                 ["Neurons are numbers in layers", "139,255 real neurons"],
                 ["Weights are learned by training on data", "15 million real synapse strengths, counted under a microscope"],
                 ["Input layer → hidden layers → output layer", "Sensory neurons → interneurons → motor neurons (same idea!)"],
                 ["Densely connected", "Sparse — each neuron talks to ~100 others"],
                 ["Mostly feed-forward", "Full of loops (recurrent)"],
                 ["Outputs a continuous number", "Fires spikes (on/off pulses) — like neuromorphic chips"],
                 ["Designed by people, trained by gradient descent", "Designed by ~500 million years of evolution"]],
                [8.25 * cm, 8.25 * cm]),
          Spacer(1, 8),
          P("So we can do the usual AI things with it:"),
          bullets(["<b>Use it as an architecture</b> — freeze the wiring, train only a small output layer, compare with random wiring.",
                   "<b>Train it</b> — keep the wiring, let the synapse strengths learn by backpropagation, compare with a CNN.",
                   "<b>Interpret it</b> — the same tools used to understand ChatGPT (ablation, probing) applied to a real brain, where biologists already know the answers, so we can check whether the tools work.",
                   "<b>Graph neural networks</b> — the brain is a huge graph; a GNN can predict missing labels (e.g. which neurons are inhibitory).",
                   "<b>LLM agents</b> — the AI scientist is a tool-using agent, the most modern kind of AI application.",
                   "<b>Energy</b> — the fly does vision + flight on ~1 milliwatt. We can count \"spikes per decision\" and compare with a GPU's operations. This is the low-power AI story."])]

story += [PageBreak()]

# ---------------- 5b. Demo ----------------
story += [P("6. How we will demo it to the teachers", H1),
          P("<b>What exists today:</b> a working animation (GIF) generated from the real simulation. Neurons are placed "
            "left to right by how many synapses they are from the tongue. Press play: sugar hits the sensors, glowing "
            "dots ripple to the right hop by hop, and within ~30 ms the feeding neuron MN9 (green circle) starts "
            "firing; a trace below counts its spikes and the label flips to \"FLY IS FEEDING\"."),
          P("<b>Phase-1 demo (Friday):</b> play the GIF, then show the two result charts (sugar vs random; cutting 3 "
            "neurons stops feeding). If there is time and a laptop is allowed, run the script live — one second of "
            "brain time takes about 35 seconds, which is fine for a 10-minute slot."),
          P("<b>Final demo (December) — the web dashboard:</b>"),
          bullets(["A map of the brain in the browser (2-D, or 3-D with real neuron shapes from FlyWire).",
                   "Buttons: <b>Sugar</b>, <b>Bitter</b>, <b>Touch leg</b>, <b>Light</b> — press one and watch neurons light up live.",
                   "A <b>scalpel</b> tool: click neurons to cut them, then press Sugar again and see feeding stop.",
                   "A <b>chat box</b> to the AI scientist: type \"what makes the fly eat?\" and watch it run experiments and answer with a graph.",
                   "A <b>benchmark tab</b>: the fly-wiring network vs a random network learning the same task, accuracy curves side by side."]),
          P("So yes — it is animated, interactive, and something a non-expert can understand in 30 seconds.")]

# ---------------- 6. Problem & goal ----------------
story += [P("7. The problem we solve and the final goal", H1),
          P("<b>The problem:</b> We have the map, but a map is not understanding. Finding out which neurons cause "
            "which behaviour still needs an expert to guess, stimulate, cut and repeat — for months. And nobody knows "
            "whether evolved wiring is a better neural-network design than human-made ones.", QUOTE),
          P("<b>Our final goal — two things we can measure:</b>"),
          bullets(["<b>Automatic circuit discovery.</b> Given a behaviour (feeding, grooming, escape), our AI scientist finds the responsible neurons on its own. Success = it rediscovers the circuits published in the Nature paper, and we report how many experiments it needed.",
                   "<b>Evolution vs. engineering.</b> A benchmark showing where the fly's wiring beats, ties or loses to random and standard networks on 3–5 learning tasks, plus the shuffle curve."]),
          P("<b>Formal problem statement (for forms and the PPT):</b>"),
          P("Whole-brain connectomes are now available, but extracting function from structure requires slow, "
            "expert-driven experimentation, and it is unknown whether biologically evolved connectivity offers "
            "advantages as a machine-learning architecture. We build a laptop-scale simulation platform for the "
            "<i>Drosophila</i> connectome, an LLM agent that autonomously designs, executes and interprets stimulation "
            "and lesion experiments to identify behavioural circuits, and a benchmark comparing the connectome as a "
            "neural-network substrate against random and engineered alternatives.", SMALL)]

# ---------------- 7. Why it matters ----------------
story += [P("8. Why it matters in real life", H1),
          table([["Area", "How our work helps"],
                 ["Neuroscience labs", "Simulate first, experiment second — turns months of lab work into a shortlist of neurons to test."],
                 ["Medicine", "Flies share ~75% of human disease genes and are standard models for Parkinson's, Alzheimer's, epilepsy, sleep and addiction. A working brain model lets researchers ask \"if these neurons fail, what breaks?\""],
                 ["AI research", "Evidence on whether to copy biological wiring into new architectures (early results say it overfits less and controls movement better)."],
                 ["Low-power computing", "The brain does complex tasks on almost no energy. Neuromorphic chips (Intel Loihi, IBM) already run this fly brain as a test program. Data centres use huge power; brains do not."],
                 ["Future brains", "The mouse connectome is being mapped now; human is the long-term goal. Tools built on the fly will be reused there."]],
                [4 * cm, 12.5 * cm])]

# ---------------- 8. Timeline ----------------
story += [P("9. Timeline (semester plan)", H1),
          table([["Phase", "When", "What we deliver"],
                 ["1. Foundation", "17 Sep – 1 Oct", "Scope agreed with mentor, GitHub repo, cleaned simulator, cell-type labels loaded"],
                 ["2. Validation + Atlas", "1 Oct – 22 Oct", "Feeding, bitter-suppression and grooming reproduced; \"we match the paper\" table; graph queries"],
                 ["3. AI scientist", "22 Oct – 19 Nov", "Tools exposed to the LLM; agent rediscovers the feeding circuit, then grooming"],
                 ["4. NN benchmark", "5 Nov – 3 Dec", "Fly wiring vs random/MLP on 3–5 tasks; shuffle curve"],
                 ["5. Dashboard + paper", "3 Dec – 17 Dec", "Web dashboard, demo, report, IEEE-format paper draft"]],
                [3.6 * cm, 3.2 * cm, 9.7 * cm])]

story += [PageBreak()]

# ---------------- 9. Q&A ----------------
story += [P("10. Questions the teacher may ask — and simple answers", H1)]
qa = [
    ("Was the fly brain released in 2024 or 2026?",
     "Both are true. The female brain (FlyWire, the one we use) was published in Nature on 3 October 2024. In June 2026 two bigger maps came out: the female brain + nerve cord (BANC, Nature) and the complete male nervous system (MaleCNS, Cell). The recent videos online are about the 2026 releases. Our project can use all of them."),
    ("Is this a biology project or a CS project?",
     "CS. The biology is a dataset we download. Everything we build is simulation, graph algorithms, machine learning, LLM agents and a web app. We need to read two papers, not study biology."),
    ("What is new? Hasn't the brain already been simulated?",
     "Yes, simulating it was done in 2024, and we reproduced that. What is new: (1) an AI agent that runs experiments on the brain autonomously — nobody has done this; (2) a systematic test of the fly's wiring against random and standard networks across many tasks — only single-task tests exist."),
    ("Can a laptop really run 139,000 neurons?",
     "Yes — we already did it. The data is 100 MB. One second of brain time takes about 40 seconds. The neurons are simple and the wiring is sparse, so it is just fast sparse-matrix maths."),
    ("Where is the data from? Is it legal to use?",
     "FlyWire (codex.flywire.ai), released publicly in Nature, October 2024. Free for research and education."),
    ("What is the real-life use?",
     "Faster neuroscience (simulate before experimenting), disease-model research, better and lower-power AI architectures, and the same tools will be needed for the mouse and human connectomes being mapped now."),
    ("What will the demo look like?",
     "A web page with the brain. Press \"sugar\" — neurons light up and the feeding neuron fires. Cut three neurons — feeding stops. Type \"what makes the fly eat?\" — the AI scientist runs experiments and answers with a graph."),
    ("Is there any hardware?",
     "Not required. Optional: a Raspberry Pi running a small circuit (the escape reflex) driving an LED or servo, if the panel wants a physical element."),
    ("What if the AI agent doesn't work?",
     "The simulator, atlas, validation table and benchmark are independent deliverables — each is a complete result on its own. The agent is the stretch that makes it special, but the project stands without it."),
    ("How is the work divided?",
     "Four modules, one per person: brain engine, brain atlas, AI scientist, benchmark + dashboard. Each has a clear interface so people can work in parallel."),
    ("What is the publication plan?",
     "An IEEE-format paper on autonomous circuit discovery and the evolved-vs-random benchmark; plus an open-source release on GitHub."),
]
for q, a in qa:
    story.append(KeepTogether([P(f"<b>Q: {q}</b>", ParagraphStyle("q", parent=BODY, textColor=NAVY, spaceAfter=2)),
                               P(f"A: {a}")]))

# ---------------- 10. Glossary ----------------
story += [P("11. Words you will hear, explained simply", H1),
          table([["Word", "Meaning"],
                 ["Neuron", "A brain cell. It receives signals, and when enough arrive it \"fires\" a pulse."],
                 ["Synapse", "The connection where one neuron passes a signal to another. Can excite (push) or inhibit (hold back)."],
                 ["Connectome", "The complete map of all neurons and all synapses in a brain."],
                 ["FlyWire", "The project and dataset that mapped the fruit-fly brain."],
                 ["Spiking neural network", "A network where neurons send on/off pulses, like real brains, instead of continuous numbers."],
                 ["Leaky integrate-and-fire", "The simple neuron model we use: it fills up with input, leaks a little, and fires when it crosses a threshold."],
                 ["Stimulate / Lesion", "Turn neurons on / remove neurons — the two basic experiments in neuroscience."],
                 ["MN9", "The motor neuron that extends the fly's mouth (proboscis) to eat. Our \"is it eating?\" indicator."],
                 ["Reservoir computing", "Using a fixed, complex network as-is and training only a small output layer on top."],
                 ["LLM agent", "A language model (like Claude) that can call tools and act in a loop, not just answer text."],
                 ["Neuromorphic chip", "Computer hardware built to run spiking networks with very low power (e.g. Intel Loihi)."]],
                [4 * cm, 12.5 * cm]),
          Spacer(1, 10),
          PageBreak(),
          P("12. References (all checked against the DOI registry on 17 Sep 2026)", H1),
          bullets([
            "Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain. <i>Nature</i>, 634, 124–138. doi:10.1038/s41586-024-07558-y",
            "Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome cell typing of Drosophila. <i>Nature</i>, 634, 139–152. doi:10.1038/s41586-024-07686-5",
            "Lin, A., et al. (2024). Network statistics of the whole-brain connectome of Drosophila. <i>Nature</i>, 634, 153–165. doi:10.1038/s41586-024-07968-y",
            "Shiu, P. K., et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. <i>Nature</i>, 634, 210–219. doi:10.1038/s41586-024-07763-9",
            "Lappalainen, J. K., et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. <i>Nature</i>, 634, 1132–1140. doi:10.1038/s41586-024-07939-3",
            "Nern, A., et al. (2025). Connectome-driven neural inventory of a complete visual system. <i>Nature</i>, 641, 1225–1237. doi:10.1038/s41586-025-08746-0",
            "Vaxenburg, R., et al. (2025). Whole-body physics simulation of fruit fly locomotion. <i>Nature</i>, 643, 1312–1320. doi:10.1038/s41586-025-09029-4",
            "Stürner, T., et al. (2025). Comparative connectomics of Drosophila descending and ascending neurons. <i>Nature</i>, 643, 158–172. doi:10.1038/s41586-025-08925-z",
            "Costi, L., Hadjiivanov, A., Dold, D., Hale, Z. F., & Izzo, D. (2025). The Drosophila connectome as a computational reservoir for time-series prediction. <i>Biomimetics</i>, 10(5), 341. doi:10.3390/biomimetics10050341",
            "Bates, A. S., Phelps, J. S., Kim, M., Yang, H. H., et al. (2026). Distributed control circuits across a brain-and-cord connectome. <i>Nature</i>. doi:10.1038/s41586-026-10735-w",
            "FlyEM Project Team et al. (2026). Sexual dimorphism in the complete connectome of the Drosophila male central nervous system. <i>Cell</i>. cell.com/cell/fulltext/S0092-8674(26)00942-6",
            "Jin, Z., Zhu, Y., Zhang, C., & Sui, Y. (2026). Whole-brain connectomic graph model enables whole-body locomotion control in fruit fly. arXiv:2602.17997",
            "State of Brain Emulation Report 2025. (2025). arXiv:2510.15745",
            "FlyWire Codex (dataset v783): codex.flywire.ai · Shiu model code: github.com/philshiu/Drosophila_brain_model · MaleCNS v1.0: male-cns.janelia.org"],
            SMALL)]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Connectome Lab — project explained simply",
                        author="EL Team, RVCE")
doc.build(story)
print("saved", OUT)
