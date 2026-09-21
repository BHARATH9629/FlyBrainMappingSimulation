#!/bin/bash
# milestone-3 compute for one sex: atlas entries (with search), batch, fidelity, sensitivity. Logs under atlas/<sex>/m3_*.log
sex=$1; cd "$(dirname "$0")"
F="grep --line-buffered -v Warning"
python -u run_redteam.py --sex $sex --behaviour eat            2>&1 | $F > atlas/$sex/m3_eat.log
python -u run_redteam.py --sex $sex --behaviour escape         2>&1 | $F > atlas/$sex/m3_escape.log
python -u run_redteam.py --sex $sex --behaviour refuse_bitter  2>&1 | $F > atlas/$sex/m3_refuse_bitter.log
python -u batch.py --sex $sex --behaviours eat escape --reps 3 2>&1 | $F > atlas/$sex/m3_batch.log
python -u fidelity.py --sex $sex --behaviour escape            2>&1 | $F > atlas/$sex/fidelity_escape.log
python -u fidelity.py --sex $sex --behaviour refuse_bitter     2>&1 | $F > atlas/$sex/fidelity_refuse_bitter.log
python -u sensitivity.py --sex $sex --behaviour escape         2>&1 | $F > atlas/$sex/sensitivity_escape.log
echo "M3 $sex DONE" > atlas/$sex/m3_done.txt
