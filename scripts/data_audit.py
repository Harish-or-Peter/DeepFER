import os, json, csv, hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import imagehash
import pandas as pd

CONFIG = {
    "splits": {
        "train": "data/train",
        "val":   "data/validation",   # change to data/val if you renamed
        "test":  "data/test"
    },
    "classes": ["angry","disgust","fear","happy","neutral","sad","surprise"],
    "out_dir": "results/metrics",
    "dup_out": "results/metrics/duplicates.csv",
    "per_class_out": "results/metrics/data_audit_per_class.csv",
    "summary_out": "results/metrics/data_audit.json",
    "valid_exts": {".jpg",".jpeg",".png",".bmp",".webp"}
}

def verify_image(p: Path):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            return True, im.size  # (w, h)
    except (UnidentifiedImageError, OSError):
        return False, None

def perceptual_hash(p: Path):
    try:
        with Image.open(p) as im:
            return str(imagehash.phash(im))
    except Exception:
        return None

def main():
    Path(CONFIG["out_dir"]).mkdir(parents=True, exist_ok=True)
    per_class_rows = []
    dups_rows = []
    summary = {
        "splits": {},
        "global": {"total":0, "corrupt":0, "non_image":0, "min_size":None, "max_size":None}
    }

    phash_map = {}  # hash -> list of paths (for duplicates)

    for split, root in CONFIG["splits"].items():
        root = Path(root)
        if not root.exists(): 
            continue
        split_tot = split_corrupt = split_nonimg = 0
        min_w=min_h=10**9
        max_w=max_h=0

        for cls in CONFIG["classes"]:
            cdir = root/cls
            if not cdir.is_dir(): 
                raise FileNotFoundError(f"Missing class folder: {cdir}")
            files = [p for p in cdir.rglob("*") if p.is_file()]
            ok_count = 0
            for p in files:
                if p.suffix.lower() not in CONFIG["valid_exts"]:
                    split_nonimg += 1
                    continue
                ok, size = verify_image(p)
                if not ok:
                    split_corrupt += 1
                    continue
                ok_count += 1
                w,h = size
                min_w, min_h = min(min_w,w), min(min_h,h)
                max_w, max_h = max(max_w,w), max(max_h,h)
                # perceptual hash for duplicate detection
                ph = perceptual_hash(p)
                if ph:
                    phash_map.setdefault(ph, []).append(str(p))

            per_class_rows.append({"split":split,"class":cls,"count":ok_count})
            split_tot += ok_count

        summary["splits"][split] = {
            "total": split_tot,
            "corrupt": split_corrupt,
            "non_image": split_nonimg,
            "min_size": [min_w if min_w!=10**9 else None, min_h if min_h!=10**9 else None],
            "max_size": [max_w if max_w!=0 else None, max_h if max_h!=0 else None]
        }
        summary["global"]["total"] += split_tot
        summary["global"]["corrupt"] += split_corrupt
        summary["global"]["non_image"] += split_nonimg
        # update global min/max
        def upd(global_key, curr):
            if curr[0] is None: return
            gw, gh = summary["global"][global_key] or [None,None]
            if gw is None:
                summary["global"][global_key] = curr
            else:
                if global_key=="min_size":
                    summary["global"][global_key] = [min(gw,curr[0]), min(gh,curr[1])]
                else:
                    summary["global"][global_key] = [max(gw,curr[0]), max(gh,curr[1])]
        upd("min_size", summary["splits"][split]["min_size"])
        upd("max_size", summary["splits"][split]["max_size"])

    # duplicates (same perceptual hash with >1 files)
    for ph, paths in phash_map.items():
        if len(paths) > 1:
            for p in paths:
                dups_rows.append({"hash": ph, "path": p, "group_size": len(paths)})

    # write per-class CSV
    df = pd.DataFrame(per_class_rows)
    if not df.empty:
        # add percentage within split
        df["split_total"] = df.groupby("split")["count"].transform("sum")
        df["pct_within_split"] = (df["count"] / df["split_total"]).round(4)
        df = df.sort_values(["split","class"])
        df.to_csv(CONFIG["per_class_out"], index=False)

    # write duplicates CSV
    if dups_rows:
        pd.DataFrame(dups_rows).sort_values(["group_size","hash"], ascending=[False,True]).to_csv(CONFIG["dup_out"], index=False)

    # write summary JSON
    with open(CONFIG["summary_out"], "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {CONFIG['per_class_out']}")
    print(f"Wrote {CONFIG['summary_out']}")
    if dups_rows:
        print(f"Wrote {CONFIG['dup_out']} (duplicates found: {len(set([r['hash'] for r in dups_rows]))} groups)")
    else:
        print("No duplicates detected by perceptual hash.")
    
    # Recommend class weights for training (useful later)
    for split in ("train",):
        sdf = df[df["split"]==split]
        if not sdf.empty:
            counts = sdf.set_index("class")["count"].to_dict()
            total = sum(counts.values())
            weights = {c: round(total/(len(counts)*n), 4) for c,n in counts.items() if n>0}
            print("Suggested class weights (train):", weights)

if __name__ == "__main__":
    main()
