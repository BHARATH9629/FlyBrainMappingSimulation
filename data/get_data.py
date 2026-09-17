"""Download the FlyWire v783 connectome files used by this project (from the Nature 2024 paper's public repo).
Run once:  python data/get_data.py
"""
import os, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
for name in ("Connectivity_783.parquet", "Completeness_783.csv"):
    dst = os.path.join(HERE, name)
    if os.path.exists(dst):
        print("already have", name); continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(BASE + name, dst)
    print("  ->", os.path.getsize(dst) // 1_000_000, "MB")
print("done")
