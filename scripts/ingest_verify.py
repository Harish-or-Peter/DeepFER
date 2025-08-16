import os, json, csv
from pathlib import Path
from PIL import Image

CONFIG = {
    "raw_train": "./data/train",
    "raw_val": "./data/val",
    "classes": ["angry","disgust","fear","happy","neutral","sad","surprise"],
    "metrics_out": "./results/metrics/data_overview.csv",
    "label_map_out": "./data/metadata/classes.json"
}

def is_image_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False

def count_split(split_dir):
    rows = []
    total = 0
    corrupt = 0
    for c in CONFIG["classes"]:
        cdir = Path(split_dir) / c
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        files = [p for p in cdir.rglob("*") if p.is_file()]
        n = 0
        for f in files:
            if is_image_ok(f):
                n += 1
            else:
                corrupt += 1
        rows.append((Path(split_dir).name, c, n))
        total += n
    return rows, total, corrupt

def main():
    os.makedirs(Path(CONFIG["metrics_out"]).parent, exist_ok=True)
    # save mapping (idempotent)
    mapping = {"label_to_index": {c:i for i,c in enumerate(CONFIG["classes"])}}
    Path(CONFIG["label_map_out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["label_map_out"], "w") as f:
        json.dump(mapping, f)

    all_rows = []
    for split in ("raw_train","raw_val"):
        rows, total, corrupt = count_split(CONFIG[split])
        all_rows.extend(rows)
        print(f"[{split}] images: {total} | corrupt: {corrupt}")

    # write CSV
    with open(CONFIG["metrics_out"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split","class","count"])
        writer.writerows(all_rows)
    print(f"Wrote {CONFIG['metrics_out']}")

if __name__ == "__main__":
    main()
