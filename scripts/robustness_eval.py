import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data")
CKPT_DIR = Path("models/checkpoints")
METRICS_DIR = Path("results/metrics"); METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("results/plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def latest_best():
    cks = sorted(CKPT_DIR.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cks[0] if cks else None

def load_labels():
    with open(DATA_DIR/"metadata"/"classes.json") as f:
        m = json.load(f)["label_to_index"]
    return [k for k,_ in sorted(m.items(), key=lambda kv: kv[1])]

def infer_backbone(run_tag: str):
    if "efficientnet_b0" in run_tag: return "efficientnet_b0"
    if "resnet50" in run_tag: return "resnet50"
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

# deterministic corruption ops
class AdjustBrightness:
    def __init__(self, factor: float): self.factor = factor
    def __call__(self, img): return TF.adjust_brightness(img, self.factor)

class FixedRotate:
    def __init__(self, degrees: float): self.deg = degrees
    def __call__(self, img): return TF.rotate(img, self.deg, fill=0)

class GaussianNoise:
    def __init__(self, sigma: float): self.sigma = sigma
    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("GaussianNoise expects a Tensor (after ToTensor)")
        noise = torch.randn_like(tensor) * self.sigma
        out = tensor + noise
        return torch.clamp(out, 0.0, 1.0)

def make_pipeline(corruption=None, param=None):
    """
    corruption: None | 'brightness' | 'blur' | 'rotation' | 'noise'
    param: numeric parameter for corruption
    """
    pre = []
    post = []
    # base eval pipeline
    pre += [transforms.Resize(256), transforms.CenterCrop(224)]

    if corruption == "brightness":
        pre += [AdjustBrightness(param)]
    elif corruption == "blur":
        # param is sigma; kernel size auto-picked
        k = 3 if param < 1.5 else 5
        pre += [transforms.GaussianBlur(kernel_size=k, sigma=param)]
    elif corruption == "rotation":
        pre += [FixedRotate(param)]

    pre += [transforms.ToTensor()]
    if corruption == "noise":
        post += [GaussianNoise(param)]
    # normalize last
    post += [transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
    return transforms.Compose(pre + post)

@torch.no_grad()
def eval_imagefolder(model, root: Path, transform, batch_size=64):
    ds = datasets.ImageFolder(root, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in dl:
        logits = model(xb.to(DEVICE))
        pred = logits.argmax(1).cpu().numpy()
        y_true.append(yb.numpy()); y_pred.append(pred)
    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    return accuracy_score(y_true, y_pred) if y_true.size else None

def plot_matrix(acc_table, severities, out_png):
    # acc_table: dict[corruption] -> list per severity
    corrs = list(acc_table.keys())
    fig, ax = plt.subplots(figsize=(8.5,4.8))
    for corr, vals in acc_table.items():
        ax.plot(severities, vals, marker="o", label=corr)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness: Accuracy vs Severity")
    ax.set_xticks(severities)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    labels = load_labels()
    num_classes = len(labels)
    ckpt = latest_best()
    assert ckpt is not None, "No *_best.pt found in models/checkpoints"
    run_tag = ckpt.stem.replace("_best","")
    backbone = infer_backbone(run_tag)
    print(f"[robust] analyzing run={run_tag} backbone={backbone}")

    model = build_model(backbone, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # roots
    val_root = DATA_DIR/"val"
    test_root = DATA_DIR/"test"
    target_root = test_root if test_root.exists() else val_root

    # clean baseline
    clean_tf = make_pipeline(None, None)
    clean_acc = eval_imagefolder(model, target_root, clean_tf)
    print(f"[robust] clean accuracy ({'test' if target_root==test_root else 'val'}): {clean_acc:.4f}")

    # corruption grid
    severities = [1,2,3]
    grid = {
        "brightness": {1:0.8, 2:0.6, 3:0.4},  # factors
        "blur":       {1:1.0, 2:1.5, 3:2.0},  # sigma
        "rotation":   {1:10,  2:20,  3:30},   # degrees
        "noise":      {1:0.03,2:0.06,3:0.10}, # stdev in [0..1]
    }

    results = {
        "run": run_tag,
        "backbone": backbone,
        "split": "test" if target_root==test_root else "val",
        "clean_acc": float(clean_acc),
        "corruptions": {}
    }

    acc_table = {}
    for corr, sevmap in grid.items():
        per_sev = []
        for sev, param in sevmap.items():
            tf = make_pipeline(corruption=corr, param=param)
            acc = eval_imagefolder(model, target_root, tf)
            per_sev.append(float(acc))
            print(f"[robust] {corr} sev{sev}({param}) -> acc={acc:.4f}")
            results["corruptions"].setdefault(corr, {})[str(sev)] = {
                "param": param, "accuracy": float(acc),
                "delta": float(acc - clean_acc)
            }
        acc_table[corr] = per_sev

    # save artifacts
    out_json = METRICS_DIR/f"{run_tag}_robustness.json"
    out_csv  = METRICS_DIR/f"{run_tag}_robustness.csv"
    with open(out_json, "w") as f: json.dump(results, f, indent=2)

    # CSV
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption","severity","param","accuracy","delta_vs_clean"])
        for corr, sevmap in results["corruptions"].items():
            for sev, info in sevmap.items():
                w.writerow([corr, sev, info["param"], info["accuracy"], info["delta"]])

    # plot
    plot_matrix(acc_table, severities, PLOTS_DIR/f"{run_tag}_robustness.png")
    print(f"[robust] saved: {out_json}, {out_csv}, {PLOTS_DIR/f'{run_tag}_robustness.png'}")

if __name__ == "__main__":
    main()
