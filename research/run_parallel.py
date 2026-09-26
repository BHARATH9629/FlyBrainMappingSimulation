"""Run several research/run_instance.py jobs at the same time, each in its OWN Python process (one core each).
Nothing inside a simulation changes: every job is exactly the single-process run, so results are identical;
only independent jobs overlap in time.

Usage: python research/run_parallel.py --workers 4 [--outdir NAME] JOB [JOB ...]     JOB = "name:goal[:unscaled]"
       python research/run_parallel.py --workers 4 --screen          (the frozen screen attack set, goal 'block')
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
W = int(a[a.index("--workers") + 1]); OUT = a[a.index("--outdir") + 1] if "--outdir" in a else "instances"
if "--screen" in a:
    sys.path.insert(0, HERE); import instances
    jobs = [(n, "block", False) for n, *_ in instances.screen_jobs()]
else:
    skip = {a.index("--workers") + 1} | ({a.index("--outdir") + 1} if "--outdir" in a else set())
    jobs = [(j.split(":")[0], j.split(":")[1], j.endswith(":unscaled")) for k, j in enumerate(a) if k not in skip and not j.startswith("--")]
logdir = os.path.join(HERE, "results", OUT); os.makedirs(logdir, exist_ok=True)


def one(job):
    n, g, uns = job; tag = f"{n}_{g}{'_unscaled' if uns else ''}"
    if os.path.exists(os.path.join(logdir, tag + ".json")): return f"{tag}: already done, skipped"
    cmd = [sys.executable, os.path.join(HERE, "run_instance.py"), n, g, "--outdir", OUT] + (["--unscaled"] if uns else [])
    t0 = time.time()
    with open(os.path.join(logdir, tag + ".log"), "w") as f: rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    last = open(os.path.join(logdir, tag + ".log")).read().strip().splitlines()
    res = [l for l in last if l.startswith(n)]
    return f"{tag}: {'exit %d' % rc if rc else (res[-1] if res else 'no result line')}  [{time.time() - t0:.0f}s]"


print(f"{len(jobs)} jobs, {W} at a time -> results/{OUT}/", flush=True); t0 = time.time()
with ThreadPoolExecutor(W) as ex:
    for k, r in enumerate(ex.map(one, jobs)): print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {r}", flush=True)
print("all done", flush=True)
