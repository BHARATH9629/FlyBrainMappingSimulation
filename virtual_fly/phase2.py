"""Phase 2 launcher: the fly and its brain, behaviours only (no red / blue team).
python virtual_fly/phase2.py [--sex female|male]
"""
import os, sys, runpy
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.py"), "--mode", "phase2"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
