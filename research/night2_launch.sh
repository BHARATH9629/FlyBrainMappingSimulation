#!/bin/bash
# Night 27 Sep (pre-registered same day): H5 transfer into each fly, then H2 (forbidden direct inputs) on the attacked screen runs.
cd "$(dirname "$0")"
python h5_transfer_screen.py FlyWire > results/h5/transfer_into_FlyWire.log 2>&1 &
python h5_transfer_screen.py male > results/h5/transfer_into_male.log 2>&1 &
python h5_transfer_screen.py BANC > results/h5/transfer_into_BANC.log 2>&1 &
python h5_transfer_screen.py BANC --scale 1.0 > results/h5/transfer_into_BANC_x1.0.log 2>&1 &
wait; echo "TRANSFERS DONE"
python run_parallel.py --workers 6 --screen --only-attacked instances --script h2_bottleneck.py --outdir h2
python run_parallel.py --workers 6 --screen --fly BANC --only-attacked sweep_BANC_x1.0 --script h2_bottleneck.py --outdir h2 --scale 1.0
echo "NIGHT DONE"
