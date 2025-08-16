import json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models, datasets, transforms

from data_preprocessing import make_loaders  # reuse your dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data")
CLASS_MAP_FILE = DATA_DIR/"metadata"/"classes.json"
CKPT_DIR = Path("models/checkpoints")
OUT_DIR = Path("results/metrics"); OUT_DIR.mkdir(parents=True, exist_ok=True)
ERR_DIR = Path("results/errors"); ERR_DIR.mkdir(parents=True, exist_ok=True)

def load_labels():
    with open(CLASS_MAP_FILE) as f:
        m = json.load(f)["label_to_index"]
    # order by index
    return [name for name,_ in sorted(m.items(), key=lambda kv: kv[1])]

def latest_best_ckpt():
    cands = sorted(CKPT_DIR.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def infer_backbone_from_name(name: str):
    if "efficientnet_b0" in name: return "efficientnet_b0"
    if "resnet50" in name: return "resnet50"
    # default to efficientnet_b0
    return "efficientnet_b0"

def build_model(backbone: str, num_classes: int):
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        return m
    elif backbone == "resnet50":
        m = models.resnet50(weights=None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

@torch.no_grad()
def eval_collect(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        y_true.append(yb.numpy())
        y_pred.append(pred)
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(y_pred)

def save_confusion(cm, labels, out_png, out_csv):
    # csv
    np.savetxt(out_csv, cm, fmt="%d", delimiter=",")
    # heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # normalized percentages for annotation
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.1f}%)", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def try_write_misclassifications(run_tag, labels, model):
    """
    Best-effort: create test dataset with ImageFolder to capture file paths.
    If your folder names differ, this will silently skip.
    """
    test_root = DATA_DIR / "test"
    if not test_root.exists():
        return
    # use a simple eval transform that should match your val/test pipeline
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    try:
        ds = datasets.ImageFolder(test_root, transform=tfm)
    except Exception:
        return

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    paths, trues, preds = [], [], []

    with torch.no_grad():
        model.eval()
        for i, (xb, yb) in enumerate(loader):
            logits = model(xb.to(DEVICE))
            pred = logits.argmax(1).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
            # map batch indices to original file paths
            start = i * loader.batch_size
            end = start + xb.size(0)
            batch_paths = [ds.samples[k][0] for k in range(start, end)]
            paths.extend(batch_paths)

    import pandas as pd
    y_true = np.concatenate(trues) if trues else np.array([])
    y_pred = np.concatenate(preds) if preds else np.array([])

    df = pd.DataFrame({
        "path": paths,
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true": [labels[i] for i in y_true] if len(y_true) else [],
        "pred": [labels[i] for i in y_pred] if len(y_pred) else [],
        "correct": (y_true == y_pred).astype(int) if len(y_true) else []
    })
    out_csv = OUT_DIR / f"{run_tag}_test_predictions.csv"
    try:
        df.to_csv(out_csv, index=False)
        # copy a small gallery of mistakes (optional)
        gallery = ERR_DIR / run_tag
        gallery.mkdir(parents=True, exist_ok=True)
        mistakes = df[df["correct"] == 0].sample(min(64, (df["correct"]==0).sum()), random_state=42) if len(df)>0 else []
        for _, row in mistakes.iterrows():
            src = Path(row["path"])
            if src.exists():
                dest = gallery / f"{src.stem}__true-{row['true']}__pred-{row['pred']}{src.suffix}"
                try:
                    dest.write_bytes(src.read_bytes())
                except Exception:
                    pass
        print(f"[miscls] CSV saved: {out_csv} | gallery: {gallery}")
    except Exception as e:
        print(f"[miscls] skipped ({e})")

def main():
    labels = load_labels()
    num_classes = len(labels)

    # dataloaders (reuse your pipeline for fair metrics)
    train_loader, val_loader, test_loader = make_loaders(batch_size=64)

    # checkpoint
    best = latest_best_ckpt()
    assert best is not None, "No *_best.pt found in models/checkpoints"
    run_tag = best.stem.replace("_best","")
    backbone = infer_backbone_from_name(run_tag)
    print(f"[analyze] run={run_tag} backbone={backbone}")

    # model
    model = build_model(backbone, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(best, map_location=DEVICE))

    # ---- evaluate on val/test ----
    yv, pv = eval_collect(model, val_loader, DEVICE)
    yt, pt = eval_collect(model, test_loader, DEVICE) if (test_loader and len(test_loader)) else (np.array([]), np.array([]))

    # reports
    val_rep = classification_report(yv, pv, target_names=labels, digits=4, zero_division=0)
    (OUT_DIR/f"{run_tag}_val_report.txt").write_text(val_rep)
    print("\n[val report]\n", val_rep)

    if yt.size:
        test_rep = classification_report(yt, pt, target_names=labels, digits=4, zero_division=0)
        (OUT_DIR/f"{run_tag}_test_report.txt").write_text(test_rep)
        print("\n[test report]\n", test_rep)

    # confusions
    cm_val = confusion_matrix(yv, pv, labels=list(range(num_classes)))
    save_confusion(cm_val, labels,
                   OUT_DIR/f"{run_tag}_val_confusion.png",
                   OUT_DIR/f"{run_tag}_val_confusion_matrix.csv")
    if yt.size:
        cm_test = confusion_matrix(yt, pt, labels=list(range(num_classes)))
        save_confusion(cm_test, labels,
                       OUT_DIR/f"{run_tag}_test_confusion.png",
                       OUT_DIR/f"{run_tag}_test_confusion_matrix.csv")
        # summary JSON
        summary = {
            "run": run_tag,
            "backbone": backbone,
            "num_classes": num_classes,
            "val_accuracy": float((yv==pv).mean()) if yv.size else None,
            "test_accuracy": float((yt==pt).mean()) if yt.size else None,
        }
        (OUT_DIR/f"{run_tag}_model_summary.json").write_text(json.dumps(summary, indent=2))

    # optional: misclassifications with file paths (best effort)
    try_write_misclassifications(run_tag, labels, model)

    print(f"[done] artifacts in: {OUT_DIR}")

if __name__ == "__main__":
    main()
