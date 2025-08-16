import argparse, json
from pathlib import Path
import torch, torch.nn as nn
from torchvision import models

DATA_DIR = Path("data")
CKPT_DIR = Path("models/checkpoints")
EXP_DIR  = Path("models/export"); EXP_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def latest_best():
    cks = sorted(CKPT_DIR.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cks[0] if cks else None

def load_labels():
    with open(DATA_DIR/"metadata"/"classes.json") as f:
        m = json.load(f)["label_to_index"]
    return [k for k,_ in sorted(m.items(), key=lambda kv: kv[1])]

def infer_backbone(name: str):
    if "efficientnet_b0" in name: return "efficientnet_b0"
    if "resnet50" in name: return "resnet50"
    return "efficientnet_b0"

def build_model(backbone: str, num_classes: int):
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
    elif backbone == "resnet50":
        m = models.resnet50(weights=None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    labels = load_labels(); k = len(labels)
    ckpt = Path(args.ckpt) if args.ckpt else latest_best()
    assert ckpt is not None, "No *_best.pt found"
    run_tag = ckpt.stem.replace("_best","")
    backbone = infer_backbone(run_tag)
    print(f"[export_ts] run={run_tag} backbone={backbone} img={args.img_size}")

    model = build_model(backbone, k).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1,3,args.img_size,args.img_size, device=DEVICE)
    ts = torch.jit.trace(model, dummy, check_trace=False)
    out_path = Path("models/export") / f"{run_tag}_ts.pt"
    ts.save(str(out_path))
    (Path("models/export") / "model_ts.pt").write_bytes(out_path.read_bytes())
    print(f"[export_ts] saved: {out_path} and models/export/model_ts.pt")

if __name__ == "__main__":
    main()
