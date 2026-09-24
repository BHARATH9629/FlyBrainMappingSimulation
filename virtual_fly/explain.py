"""Plain-language explanations for the live window: what you did, what the brain did, what it means.

Every sentence is built from the numbers of the run that just happened (rates, neuron counts, thresholds) -- nothing
here is a canned result. live.py calls explain_key() right after a key press and explain_result() after every brain
result; both return (did, brain, means). score_lines() builds the red / blue scoreboard.
"""

NAMES = {"eat": "eating", "drink": "drinking", "refuse_bitter": "refusing bitter food", "escape": "escaping",
         "retreat": "backing away", "turn": "turning away from a threat"}
TEST_KEY = {"eat": "S", "drink": "V", "refuse_bitter": "B", "escape": "L", "retreat": "DOWN", "turn": "2"}
OUT = {"eat": "the feeding neuron MN9", "drink": "the feeding neuron MN9", "refuse_bitter": "the feeding neuron MN9",
       "escape": "the giant fibre DNp01 (the escape neuron)", "retreat": "the moonwalker neurons MDN (walking backwards)",
       "turn": "the steering neuron DNa02"}
SHORT = {"eat": "MN9", "drink": "MN9", "refuse_bitter": "MN9", "escape": "giant fibre", "retreat": "MDN", "turn": "DNa02"}


def explain_key(key, mode, has_attack=False, has_repair=False):
    """(did, brain, means) right after a key press, before the brain has answered."""
    wait = "thinking ..."
    table = {
        "s": ("You put a drop of sugar in front of the fly.", "The fly walks to it; when it tastes it, its sugar neurons will fire.", ""),
        "v": ("You put a drop of water in front of the fly.", "The fly walks to it; when it tastes it, its water neurons will fire.", ""),
        "b": ("You put a drop of sugar mixed with bitter poison in front of the fly.",
              "The fly walks to it; when it tastes it, both sugar and bitter neurons will fire.", ""),
        "l": ("You showed the fly a looming shadow -- something coming straight at it.", wait, ""),
        "down": ("You showed the fly an object approaching head-on, seen by both eyes.", wait, ""),
        "2": ("You showed a threat to the fly's RIGHT eye.", wait, ""),
        "1": ("You showed a threat to the fly's LEFT eye.", wait,
              "Note: in this fly's wiring the left-eye pathway to the steering neurons is weak, so it often does not turn."),
        "w": ("You toggled walking.", "", ""), "space": ("You stopped the fly.", "", ""),
        "left": ("You steered the fly left.", "", ""), "right": ("You steered the fly right.", "", ""),
        "e": ("You toggled explore mode.", "", "The fly wanders; click the left panel to place food and it will find it."),
        "c": ("You set up the choice arena: sugar, bitter and water around the fly.", "", "The fly will visit each drop and decide."),
        "o": ("You toggled the slow spin of the brain view.", "", "Only the view turns; the brain itself is not changed."),
        "0": ("You reset the brain view to the front.", "", "Only the view changes; the brain itself is not changed."),
        "tab": ("You toggled the technical log.", "", ""),
        "t": ("QUICK TEST: the same test, with the stimulus put straight in front of the fly -- no walking.", wait, ""),
    }
    if key in table: return table[key]
    if mode == "phase2" and key in ("a", "d", "f", "h", "r"):
        return ("That key belongs to Phase 3 (the red team / blue team demo).", "", "Run  python virtual_fly/phase3.py  to use it.")
    if key == "a": return ("You sent in the RED TEAM.", "It is searching for the smallest damage that switches the fly's current behaviour off ...", "")
    if key == "f": return ("You sent in the RED TEAM with a hijack.", "It is searching for the fewest neurons that make the fly try to eat with NO food ...", "")
    if key == "h": return ("You checked for a hijack: no food at all.", wait, "")
    if key == "d":
        if not has_attack: return ("You asked the BLUE TEAM to repair.", "Nothing is broken: the brain is healthy.", "Attack it first with A (or F).")
        if has_repair: return ("You asked the BLUE TEAM to repair again.", "It is already repaired.", "Test it, attack again with A, or reset with R.")
        return ("You sent in the BLUE TEAM.", "It is searching for the cheapest repair that brings the behaviour back ...", "")
    if key == "r": return ("You reset the brain.", "Every attack and every repair has been removed.", "The brain is back to the real, healthy connectome.")
    return ("", "", "")


def _attack_words(best, name, goal):
    iv = best["intervention"]; n = best["size"]
    if "inject" in iv:
        what = f"forced {n} neuron{'s' if n > 1 else ''} to fire nonstop at 150 spikes per second"
        why = ("they are brakes on the pathway, so jamming them on shuts it" if goal == "block"
               else "they drive the feeding neuron, so jamming them on makes the fly eat")
    elif "lesion" in iv:
        what = f"cut {n} neuron{'s' if n > 1 else ''} out of the circuit"
        why = ("they carried the signal" if goal == "block"
               else "they were the brake that makes the fly refuse bitter")
    else:
        what = f"weakened {n} neuron{'s' if n > 1 else ''}"; why = "they carried the signal"
    return what, why


def explain_result(kind, d, ctx):
    """(did, brain, means) after a brain result. ctx: name, verdict, attacked (bool), healthy (Hz or None),
    stim_count, fired (neurons that fired), on_thr, off_thr."""
    name = ctx.get("name"); out = OUT.get(name, "the output neuron"); short = SHORT.get(name, "output")
    if kind == "measured":
        hz, fired, stim = d["hz"], ctx.get("fired"), ctx.get("stim_count")
        brain = (f"{stim} {'taste' if name in ('eat', 'drink', 'refuse_bitter') else 'sensory'} neurons fired" if stim else "")
        brain += (f"; {fired:,} neurons lit up across the brain" if fired else "") + f"; {out} fired at {hz:.0f} spikes per second."
        v = ctx.get("verdict") or ""; on, off = ctx["on_thr"], ctx["off_thr"]
        if not d.get("with_stim"):                           # the hijack test: no food at all
            means = (f"{short} fires with NO food ({hz:.0f} Hz, bar {on:.0f}): the hijack works -- the fly tries to eat nothing."
                     if hz >= on else f"{short} stays quiet with no food: no hijack.")
            return ("You checked for a hijack: no food at all.", brain.lstrip("; "), means)
        if name == "refuse_bitter":
            means = (f"{short} fires even though bitter is present ({hz:.0f} Hz >= {on:.0f}): the fly would swallow poison."
                     if hz >= on else f"Bitter keeps {short} quiet ({hz:.0f} Hz <= {off:.0f}): the fly refuses the poison."
                     if hz <= off else f"{short} is half-on ({hz:.0f} Hz, between {off:.0f} and {on:.0f}): the refusal is partly broken.")
        elif hz >= on:
            means = f"{short} is above its {on:.0f} Hz bar, so the fly is {NAMES[name]}."
        else:
            means = f"{short} is below its {on:.0f} Hz bar, so the fly is NOT {NAMES[name]}." + ("  The red team's attack is working." if ctx.get("attacked") else "")
        return (None, brain.lstrip("; "), means)
    if kind == "threat":
        L, R = d["dna_L"], d["dna_R"]; diff = abs(L - R); on = ctx["on_thr"]
        brain = f"Steering neurons: DNa02 left {L:.0f} Hz, right {R:.0f} Hz -- a difference of {diff:.0f} Hz."
        if diff >= on:
            side = "left" if L > R else "right"; away = (d["side"] == "R") == (L > R)
            means = f"The difference is above the {on:.0f} Hz bar, so the fly turns {side}" + (" -- away from the threat." if away else " -- INTO the threat: the brain is corrupted.")
        else:
            means = f"The difference is below the {on:.0f} Hz bar, so the fly does not turn." + ("  The red team's attack is working." if ctx.get("attacked") else "")
        return (None, brain, means)
    if kind == "attacked":
        best = d.get("best")
        if not best: return ("RED TEAM: no attack found.", "Within its budget it found no damage that switches this behaviour off.", "")
        what, why = _attack_words(best, name, d.get("goal"))
        return (f"The RED TEAM attacked {NAMES.get(name, name)}.", f"It {what} -- {why}.",
                f"Red rings show those neurons. Press {TEST_KEY.get(name, 'S')} to test the fly, or T for a quick test with no walking.")
    if kind == "replay":
        stage = d["stage"]; hz = d["hz"]; h = ctx.get("healthy")
        pct = f" ({hz / h * 100:.0f}% of healthy)" if h else ""
        brain = f"After the {stage}, {out} fires at {hz:.0f} Hz{pct}; {d['n_silenced']:,} neurons that fired in the healthy brain are now silent (pink rings)."
        return (None, brain, None)
    if kind == "repaired":
        rep = d.get("rep")
        if not rep: return ("BLUE TEAM: no repair found.", "Within its budget it found no repair.", "")
        if rep.get("partial"):
            if name == "refuse_bitter":
                means = ("The blue team LOSES this one: every repair that makes the fly refuse bitter again also stops it eating "
                         "sugar. The neurons the attacker cut were the only bitter-specific brake; nothing else can take their place.")
            else:
                means = "The blue team LOSES this one: " + rep.get("why", "no repair restores this behaviour.")
            return ("The BLUE TEAM tried to repair it.", "No repair works without breaking something else.", means)
        mv = rep.get("move", ""); parts = []
        if "bias" in mv: parts.append("made surviving neurons easier to fire")
        if "potentiate" in mv: parts.append(f"strengthened their connections to {short}")
        if "stop listening" in mv: parts.append(f"turned down the attacker's wires into {short}")
        if not parts: parts.append(mv)
        rec = rep.get("recovery")
        lvl = (" It could not reach its target (80% of healthy on each of 8 test runs), so this is the strongest repair it found."
               if rep.get("below_target") else " It checked this repair on 8 separate test runs before accepting it.")
        return (f"The BLUE TEAM repaired {NAMES.get(name, name)} using {rep['size']} neurons.",
                "It " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + ".",
                f"Blue rings show them.{lvl} Press {TEST_KEY.get(name, 'S')} to test the fly, or T for a quick test with no walking.")
    if kind == "reset": return ("You reset the brain.", "Every attack and every repair has been removed.", "The brain is back to the real, healthy connectome.")
    return None


def score_lines(sc):
    """Scoreboard for the brain panel: [(text, rgb), ...]. sc = dict(name, goal, healthy, red=(size, hz),
    blue=(size, hz) | 'lost' | None)."""
    if not sc or not sc.get("red"): return []
    name, goal, h = sc["name"], sc.get("goal"), sc.get("healthy") or 0.0
    what = NAMES.get(name, name); short = SHORT.get(name, "output")
    def level(hz):
        if goal == "force": return f"{short} {hz:.0f} Hz with NO food"
        if name == "refuse_bitter": return f"{short} {hz:.0f} Hz with bitter" + ("  (eats poison)" if hz >= 40 else "  (refuses)" if hz <= 10 else "")
        return f"{short} {hz:.0f} Hz ({hz / h * 100:.0f}% of healthy)" if h else f"{short} {hz:.0f} Hz"
    n, hz = sc["red"]; lines = [(f"RED TEAM:  {n} neuron{'s' if n != 1 else ''}  ->  {what}: {level(hz)}", (255, 110, 100))]
    if sc.get("blue") == "lost": lines.append(("BLUE TEAM:  cannot repair this one", (150, 170, 255)))
    elif sc.get("blue"):
        n, hz = sc["blue"]; lines.append((f"BLUE TEAM:  {n} neuron{'s' if n != 1 else ''}  ->  {what}: {level(hz)}", (120, 170, 255)))
    return lines
