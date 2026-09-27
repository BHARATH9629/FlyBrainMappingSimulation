"""Run several research/run_instance.py jobs at the same time, each in its OWN Python process (one core each).
Nothing inside a simulation changes: every job is exactly the single-process run, so results are identical;
only independent jobs overlap in time.

Usage: python research/run_parallel.py --workers 4 [--outdir NAME] JOB [JOB ...]     JOB = "name:goal[:unscaled]"
       python research/run_parallel.py --workers 4 --screen          (the frozen screen attack set, goal 'block')
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
W = int(a[a.index("--workers") + 1]); SCRIPT = a[a.index("--script") + 1] if "--script" in a else "run_instance.py"; OUT = a[a.index("--outdir") + 1] if "--outdir" in a else "instances"
if "--screen" in a:
    sys.path.insert(0, HERE); import instances
    fl = a[a.index("--fly") + 1] if "--fly" in a else None
    jobs = [(n, "block", False) for n, f, *_ in instances.screen_jobs() if fl in (None, f)]
    if "--only-attacked" in a:                 # H2: only fly-runs where the baseline found an attack
        src = a[a.index("--only-attacked") + 1]
        jobs = [j for j in jobs if (lambda p: os.path.exists(p) and json.load(open(p)).get("attack"))(os.path.join(HERE, "results", src, j[0] + "_block.json"))]
else:
    skip = {a.index(k) + 1 for k in ("--workers", "--outdir", "--scale", "--script", "--fly", "--only-attacked") if k in a}
    jobs = [(j.split(":")[0], j.split(":")[1], j.endswith(":unscaled")) for k, j in enumerate(a) if k not in skip and not j.startswith("--")]
logdir = os.path.join(HERE, "results", OUT); os.makedirs(logdir, exist_ok=True)


def one(job):
    n, g, uns = job; tag = f"{n}_{g}{'_unscaled' if uns else ''}"
    done = os.path.join(logdir, tag + (("_x" + a[a.index("--scale") + 1]) if "--scale" in a and SCRIPT != "run_instance.py" else "") + ".json")
    if os.path.exists(done): return f"{tag}: already done, skipped"
    cmd = [sys.executable, os.path.join(HERE, SCRIPT), n, g] + (["--outdir", OUT] if SCRIPT == "run_instance.py" else []) + (["--unscaled"] if uns else []) + (["--scale", a[a.index("--scale") + 1]] if "--scale" in a else [])
    t0 = time.time()
    tag += ("_x" + a[a.index("--scale") + 1]) if "--scale" in a and SCRIPT != "run_instance.py" else ""
    with open(os.path.join(logdir, tag + ".log"), "w") as f: rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    last = open(os.path.join(logdir, tag + ".log")).read().strip().splitlines()
    res = [l for l in last if l.startswith(n)]
    return f"{tag}: {'exit %d' % rc if rc else (res[-1] if res else 'no result line')}  [{time.time() - t0:.0f}s]"


print(f"{len(jobs)} jobs, {W} at a time -> results/{OUT}/", flush=True); t0 = time.time()
with ThreadPoolExecutor(W) as ex:
    for k, r in enumerate(ex.map(one, jobs)): print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {r}", flush=True)
print("all done", flush=True)
