import csv, os, shutil
from pathlib import Path
from PIL import Image

DUPS_CSV = "results/metrics/duplicates.csv"
QUAR = Path("data/_quarantine")
ROOTS = {"train":"data/train","val":"data/validation","test":"data/test"}  # adjust if you renamed 'validation'→'val'

def img_size(p):
    try:
        with Image.open(p) as im:
            return im.size[0]*im.size[1]
    except: return -1

def split_of(path: Path):
    p = str(path).replace("\\","/")
    for s,root in ROOTS.items():
        if f"{root}/" in p:
            return s
    return "other"

def main():
    QUAR.mkdir(parents=True, exist_ok=True)
    by_hash = {}
    with open(DUPS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            by_hash.setdefault(row["hash"], []).append(Path(row["path"]))
    moved = 0
    for h, paths in by_hash.items():
        if len(paths) < 2: 
            continue
        # pick keeper
        paths = [p for p in paths if p.exists()]
        if not paths: 
            continue
        # priority: train > val > test; then largest area
        order = {"train":0,"val":1,"test":2,"other":3}
        keeper = sorted(paths, key=lambda p:(order.get(split_of(p),3), -img_size(p)))[0]
        for p in paths:
            if p == keeper: 
                continue
            dest = QUAR / p.name
            i = 1
            while dest.exists():
                dest = QUAR / f"{dest.stem}_{i}{dest.suffix}"
                i += 1
            shutil.move(str(p), str(dest))
            moved += 1
    print(f"Moved {moved} duplicate files to {QUAR}")

if __name__ == "__main__":
    main()
