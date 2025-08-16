import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import onnxruntime as ort
from sklearn.metrics import accuracy_score

DATA_DIR = Path("data")
CKPT_DIR = Path("models/checkpoints")
METRICS_DIR = Path("results/metrics"); METRICS_DIR.mkdir(parents=True, exist_ok=True)
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

def make_loader(split="test", img_size=224, max_images=0):
    tfm = transforms.Compose([
        transforms.Resize(256 if img_size==224 else int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    root = DATA_DIR / split
    ds = datasets.ImageFolder(root, transform=tfm)
    if max_images and max_images < len(ds):
        ds.samples = ds.samples[:max_images]
    return torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="models/export/model.onnx")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_images", type=int, default=500)
    args = ap.parse_args()

    labels = load_labels(); k = len(labels)
    ckpt = Path(args.ckpt) if args.ckpt else latest_best()
    assert ckpt is not None, "No *_best.pt found"
    run_tag = ckpt.stem.replace("_best","")
    backbone = infer_backbone(run_tag)

    # torch model
    model = build_model(backbone, k).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # onnx runtime
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    iname = sess.get_inputs()[0].name

    dl = make_loader(args.split, img_size=args.img_size, max_images=args.max_images)
    y_true, y_pt, y_ort = [], [], []
    diffs = []

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            logits_pt = model(xb).cpu().numpy()
            preds_pt = logits_pt.argmax(1)

            logits_ort = sess.run(None, {iname: xb.cpu().numpy().astype(np.float32)})[0]
            preds_ort = logits_ort.argmax(1)

            y_true.append(yb.numpy())
            y_pt.append(preds_pt)
            y_ort.append(preds_ort)

            diffs.append(np.mean(np.abs(torch.softmax(torch.from_numpy(logits_pt),1).numpy()
                                        - torch.softmax(torch.from_numpy(logits_ort),1).numpy())))

    y_true = np.concatenate(y_true); y_pt = np.concatenate(y_pt); y_ort = np.concatenate(y_ort)
    acc_pt = accuracy_score(y_true, y_pt)
    acc_ort = accuracy_score(y_true, y_ort)
    mad = float(np.mean(diffs))

    out = {
        "run": run_tag,
        "split": args.split,
        "acc_pytorch": float(acc_pt),
        "acc_onnx": float(acc_ort),
        "mean_abs_prob_diff": mad,
        "n": int(len(y_true))
    }
    out_path = METRICS_DIR / f"{run_tag}_onnx_parity.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("[parity]", out)

if __name__ == "__main__":
    main()
