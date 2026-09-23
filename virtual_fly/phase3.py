"""Phase 3 launcher: the red team / blue team demo (attack and repair), kept separate from the Phase 2 behaviour demo.

Phase 2 (`python virtual_fly/phase2.py`) shows the six behaviours only and refuses every attack/repair key.
Phase 3 (this file) is where the red and blue teams live.

  trigger a behaviour first, then attack it:
    S sugar drop      -> eat            B bitter drop  -> refuse bitter     V water drop -> drink
    L looming shadow  -> escape         DOWN approach  -> retreat           1 / 2 threat -> turn
  then:
    A  RED TEAM   attack whatever the fly is doing now (block, or corrupt for bitter refusal)
    D  BLUE TEAM  repair that same behaviour
    F  RED TEAM   hijack: make the fly eat with NO sugar        H  test the hijack
    R  reset the brain (every attack and repair removed)

python virtual_fly/phase3.py [--sex female|male] [--robust] [--no_cache]
"""
import os, sys, runpy
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.py"), "--mode", "phase3"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
