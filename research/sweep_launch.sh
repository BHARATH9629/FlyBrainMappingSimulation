#!/bin/bash
# Scaling sweep (pre-registered 27 Sep, item a): parity check of the --scale option first, then the sweep.
cd "$(dirname "$0")"
python run_instance.py s_15_BANC block --outdir parity_scale --scale 1.8148104642341483 > results/parity_scale_banc.log 2>&1 &
python run_instance.py s_09_male block --outdir parity_scale --scale 0.556287170773152 > results/parity_scale_male.log 2>&1 &
wait
python - <<'PY' || { echo "PARITY FAILED - sweep not launched"; exit 1; }
import json, sys
ok = True
for t in ("s_15_BANC_block", "s_09_male_block"):
    a = json.load(open(f"results/instances/{t}.json")); b = json.load(open(f"results/parity_scale/{t}.json"))
    inc = b.pop("inclusion"); [d.pop("seconds") for d in (a, b)]
    same = a == b and inc["passes"] == 8; ok &= same
    print(t, "IDENTICAL" if same else [k for k in set(a) | set(b) if a.get(k) != b.get(k)], "inclusion", inc["passes"])
sys.exit(0 if ok else 1)
PY
python run_parallel.py --workers 6 --screen --fly BANC --scale 1.0 --outdir sweep_BANC_x1.0
python run_parallel.py --workers 6 --screen --fly male --scale 1.0 --outdir sweep_male_x1.0
python run_parallel.py --workers 6 --screen --fly BANC --scale 1.4 --outdir sweep_BANC_x1.4
echo "SWEEP DONE"
