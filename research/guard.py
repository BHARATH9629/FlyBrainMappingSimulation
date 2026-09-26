"""Safety guard for every Phase 4 research script: the demo and the Phase 3 results must never change.

  check_protected()  compares 132 protected files (demo code, the 3 core files the demo's saved attacks depend on,
                     the saved attacks themselves, the Phase 3 results, the fly data) against research/
                     PROTECTED_MANIFEST.json, and stops the script if any of them differs.
  results_dir(name)  the only place research output goes: research/results/<name>/ (never neuroredteam/atlas/).
  power_ok()         warns if the laptop is on battery (Windows slows the CPU; a sleep can stall a run for an hour).
  start(name)        all three, for the top of a research script; call finish() at the end to check again.

Usage in a research script:
    from guard import start, finish
    out = start("h1_redundancy")        # stops here if anything protected has changed
    ...  write only inside `out`  ...
    finish()
"""
import hashlib, json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MANIFEST = os.path.join(ROOT, "research", "PROTECTED_MANIFEST.json")
CORE = ["neuroredteam/agents.py", "neuroredteam/behaviours.py", "neuroredteam/brain.py"]
sys.path.insert(0, os.path.join(ROOT, "neuroredteam"))


class ProtectedFileChanged(RuntimeError):
    pass


def code_version():
    """the demo's cache key, computed exactly as virtual_fly/live.py computes it"""
    return hashlib.md5(b"".join(open(os.path.join(ROOT, f), "rb").read() for f in CORE)).hexdigest()[:8]


def check_protected(verbose=True):
    m = json.load(open(MANIFEST))
    if code_version() != m["code_version"]:
        raise ProtectedFileChanged(f"core files changed: demo cache key {code_version()} != {m['code_version']}")
    bad = [p for p, h in m["files"].items()
           if not os.path.exists(os.path.join(ROOT, p)) or hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() != h]
    if bad:
        raise ProtectedFileChanged(f"{len(bad)} protected file(s) changed or missing, e.g. {bad[:5]}")
    if verbose: print(f"[guard] {len(m['files'])} protected files unchanged; demo cache key {m['code_version']}", flush=True)


def results_dir(name):
    d = os.path.join(ROOT, "research", "results", name); os.makedirs(d, exist_ok=True)
    return d


def power_ok():
    if sys.platform != "win32": return True
    try:
        s = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        if s == "1": print("[guard] WARNING: on battery -- plug in the charger for long runs", flush=True); return False
    except Exception: pass
    return True


def start(name):
    check_protected(); power_ok()
    return results_dir(name)


def finish():
    check_protected()
    print("[guard] finished: nothing protected was touched", flush=True)


if __name__ == "__main__":
    check_protected(); power_ok()
