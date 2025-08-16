import os, json, time, random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

# ---------------- Config (edit if needed) ----------------
DATA_DIR = Path("data")
SPLITS = {"train":"train","val":"val","test":"test"}
CLASS_MAP_FILE = DATA_DIR/"metadata"/"classes.json"
AUDIT_CSV = Path("results/metrics/data_audit_per_class.csv")

OUT_DIR = Path("results")
CKPT_DIR = Path("models/checkpoints")
RUN_NAME = f"baseline_cnn_{int(time.time())}"

IMAGE_SIZE = 224
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 15
WEIGHT_DECAY = 1e-4
PATIENCE = 4                      # early stopping on val acc
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = min(8, os.cpu_count() or 2)
MIXED_PREC = torch.cuda.is_available()

# ---------------- Reproducibility ----------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster on images
seed_everything(42)

# ---------------- Data ----------------
from data_preprocessing import make_loaders

# ---------------- Model ----------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def load_class_map() -> Dict[str,int]:
    with open(CLASS_MAP_FILE) as f:
        return json.load(f)["label_to_index"]

def load_weights_from_audit(label_to_index: Dict[str,int]):
    import pandas as pd
    df = pd.read_csv(AUDIT_CSV)
    train_df = df[df["split"]=="train"].set_index("class")["count"].to_dict()
    total = sum(train_df.values()); k = len(train_df)
    w = {c: total/(k*train_df[c]) for c in train_df if train_df[c] > 0}
    weights = torch.ones(len(label_to_index), dtype=torch.float32)
    for cls, idx in label_to_index.items():
        weights[idx] = float(w.get(cls, 1.0))
    return weights

def compute_class_weights_from_loader(train_loader, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_loader:
        for i in range(num_classes):
            counts[i] += (y==i).sum().item()
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    losses = []
    all_y, all_p = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        losses.append(loss.item())
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        all_y.append(yb.cpu().numpy())
        all_p.append(preds.cpu().numpy())
    acc = correct/total if total else 0.0
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])
    return float(np.mean(losses)), acc, y_true, y_pred

def main():
    label_to_index = load_class_map()
    num_classes = len(label_to_index)
    train_loader, val_loader, test_loader = make_loaders(batch_size=BATCH_SIZE)

    model = SmallCNN(num_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # class weights
    try:
        class_weights = load_weights_from_audit(label_to_index).to(DEVICE)
        print("Using class weights from data_audit_per_class.csv")
    except Exception as e:
        print("Could not read audit weights, computing from loader:", e)
        class_weights = compute_class_weights_from_loader(train_loader, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR/"runs"/RUN_NAME
    SummaryWriter(str(run_dir)).close()  # create folder
    writer = SummaryWriter(str(run_dir))

    best_val_acc = 0.0
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PREC)

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=MIXED_PREC):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            preds = logits.argmax(1)
            running_loss += loss.item() * yb.size(0)
            running_correct += (preds==yb).sum().item()
            seen += yb.size(0)

        opt.step()
        sched.step()

        train_loss = running_loss/seen
        train_acc = running_correct/seen
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, DEVICE)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", sched.get_last_lr()[0], epoch)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        # checkpoint (each epoch)
        ckpt_path = CKPT_DIR/f"{RUN_NAME}_epoch{epoch:02d}.pt"
        torch.save({"epoch":epoch,"model":model.state_dict(),
                    "opt":opt.state_dict(),"acc":val_acc}, ckpt_path)

        # early stopping on val_acc
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR/f"{RUN_NAME}_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    # ----- Final evaluation & reports -----
    report_dir = OUT_DIR/"metrics"
    report_dir.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR/f"{RUN_NAME}_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # labels in index order
    labels = [k for k,_ in sorted(label_to_index.items(), key=lambda x: x[1])]

    # val report
    _, val_acc, yv, pv = evaluate(model, val_loader, DEVICE)
    rep = classification_report(yv, pv, target_names=labels, digits=4, zero_division=0)
    (report_dir/f"{RUN_NAME}_val_report.txt").write_text(rep)
    np.savetxt(report_dir/f"{RUN_NAME}_val_confusion_matrix.csv",
               confusion_matrix(yv, pv), fmt="%d", delimiter=",")
    print(f"Val accuracy (best ckpt): {val_acc:.4f}")

    # test report (if present)
    if test_loader is not None and len(test_loader.dataset) > 0:
        _, test_acc, yt, pt = evaluate(model, test_loader, DEVICE)
        rep_t = classification_report(yt, pt, target_names=labels, digits=4, zero_division=0)
        (report_dir/f"{RUN_NAME}_test_report.txt").write_text(rep_t)
        np.savetxt(report_dir/f"{RUN_NAME}_test_confusion_matrix.csv",
                   confusion_matrix(yt, pt), fmt="%d", delimiter=",")
        print(f"Test accuracy: {test_acc:.4f}")

    print("Reports saved to:", report_dir)

if __name__ == "__main__":
    main()
