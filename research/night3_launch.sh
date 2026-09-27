#!/bin/bash
# Robustness controls (pre-registered 27 Sep): A+C on all 65 attacked fly-runs, then B (4 model variants x 16 fly-runs).
cd "$(dirname "$0")"
python run_parallel.py --workers 6 --screen --only-attacked instances --script robust_ac.py --outdir robust
for P in W_SYN=0.23375 W_SYN=0.31625 V_TH=-46 V_TH=-44; do
  python run_parallel.py --workers 6 --screen --subset --param $P --outdir "robust_B_${P}"
done
echo "NIGHT3 DONE"
