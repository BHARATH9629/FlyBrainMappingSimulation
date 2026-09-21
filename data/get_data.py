"""Download the FlyWire v783 connectome files used by this project (from the Nature 2024 paper's public repo).
Run once:  python data/get_data.py            (add --male for the 1.1 GB male CNS too)
Female fly -> data/female_flywire/   Male fly -> data/male_malecns/
"""
import os, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
FILES = {"Connectivity_783.parquet": BASE + "Connectivity_783.parquet", "Completeness_783.csv": BASE + "Completeness_783.csv",
         # neuron annotations with positions and cell types (Schlegel et al. 2024), used to draw the brain
         "flywire_annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"}
for name, url in FILES.items():
    os.makedirs(os.path.join(HERE, "female_flywire"), exist_ok=True)
    dst = os.path.join(HERE, "female_flywire", name)
    if os.path.exists(dst):
        print("already have", name); continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(url, dst)
    print("  ->", os.path.getsize(dst) // 1_000_000, "MB")

# MaleCNS v1.0 (male brain + nerve cord; Janelia / Google, Sep 2026; CC-BY). ~1.1 GB. Only needed for --sex male.
import sys
if "--male" in sys.argv:
    MB = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    os.makedirs(os.path.join(HERE, "male_malecns"), exist_ok=True)
    for name in ("body-annotations-male-cns-v1.0-minconf-0.5.feather", "body-neurotransmitters-male-cns-v1.0.feather",
                 "connectome-weights-male-cns-v1.0-minconf-0.5.feather"):
        dst = os.path.join(HERE, "male_malecns", name)
        if os.path.exists(dst):
            print("already have", name); continue
        print("downloading", name, "...")
        urllib.request.urlretrieve(MB + name, dst)
        print("  ->", os.path.getsize(dst) // 1_000_000, "MB")
print("done")
