import os, json, time, argparse, random, traceback, sys
from pathlib import Path
from typing import Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from data_preprocessing import make_loaders

# ---------------- Args ----------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="efficientnet_b0",
                    choices=["efficientnet_b0","resnet50"])
    ap.add_argument("--epochs_head", type=int, default=3)           # head warm-up (partial unfreeze)
    ap.add_argument("--epochs_ft", type=int, default=20)            # fine-tune stage
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=3e-4)            # slightly lower for stability
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--save_all_epochs", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--use_mixup", action="store_true")             # used only in fine-tune stage
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--use_tta", action="store_true")               # eval-time hflip ensembling
    # NEW: resume support
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a *_last.pt checkpoint to resume from")
    return ap.parse_args()

# ---------------- Repro ----------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ---------------- Paths ----------------
DATA_DIR = Path("data")
CLASS_MAP_FILE = DATA_DIR/"metadata"/"classes.json"
AUDIT_CSV = Path("results/metrics/data_audit_per_class.csv")
OUT_DIR = Path("results"); (OUT_DIR/"logs").mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("models/checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PREC = torch.cuda.is_available()

# ---------------- Utils ----------------
def load_class_map() -> Dict[str,int]:
    with open(CLASS_MAP_FILE) as f:
        return json.load(f)["label_to_index"]

def class_weights_from_audit(label_to_index: Dict[str,int]):
    import pandas as pd
    df = pd.read_csv(AUDIT_CSV)
    train_df = df[df["split"]=="train"].set_index("class")["count"].to_dict()
    total = sum(train_df.values()); k = len(train_df)
    w = {c: total/(k*train_df[c]) for c in train_df if train_df[c] > 0}
    weights = torch.ones(len(label_to_index), dtype=torch.float32)
    for cls, idx in label_to_index.items():
        weights[idx] = float(w.get(cls, 1.0))
    return weights

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

@torch.no_grad()
def evaluate(model, loader, device, use_tta=False):
    model.eval()
    total, correct = 0, 0; losses = []
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if use_tta:
            logits = (logits + model(torch.flip(xb, dims=[3]))) / 2.0  # hflip TTA
        losses.append(F.cross_entropy(logits, yb).item())
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        ys.append(yb.cpu().numpy()); ps.append(pred.cpu().numpy())
    acc = correct/total if total else 0.0
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    return float(np.mean(losses)), acc, y_true, y_pred

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# NEW: resumable last checkpoint
def save_last_ckpt(path, epoch_idx, model, optimizer, scheduler, run_name, stage, backbone, num_classes):
    torch.save({
        "epoch": int(epoch_idx),                 # zero-based within the current stage
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "run": run_name,
        "stage": stage,                          # "head" or "ft"
        "backbone": backbone,
        "num_classes": num_classes,
    }, path)

# ---------------- Models ----------------
def build_model(backbone_name: str, num_classes: int):
    from torchvision import models

    if backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        # bump dropout in head for regularization
        if hasattr(model.classifier[0], "p"):
            model.classifier[0] = nn.Dropout(p=0.4)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
        # fine-tune last three feature blocks + classifier
        backbone_layers_to_unfreeze = ["features.5", "features.6", "features.7", "classifier"]

    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        backbone_layers_to_unfreeze = ["layer4", "fc"]  # last block + head

    else:
        raise ValueError("Unsupported backbone")

    return model, backbone_layers_to_unfreeze

def set_trainable(model: nn.Module, trainable: bool = False):
    for p in model.parameters():
        p.requires_grad = trainable

def unfreeze_selected(model: nn.Module, layer_name_prefixes):
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in layer_name_prefixes):
            p.requires_grad = True

# ---------------- Train helpers ----------------
def train_epoch(model, loader, criterion, optimizer, scaler, use_mixup=False, mixup_alpha=0.2):
    model.train()
    tloss, tcorrect, tcount = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            xb, y_a, y_b, lam = mixup_data(xb, yb, alpha=mixup_alpha)
        else:
            y_a = y_b = yb; lam = 1.0

        with torch.amp.autocast("cuda", enabled=MIXED_PREC):
            logits = model(xb)
            loss = lam*criterion(logits, y_a) + (1-lam)*criterion(logits, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(1)
        tcorrect += (preds == yb).sum().item()   # vs true labels
        tcount += yb.size(0)
        tloss += loss.item() * yb.size(0)

    return (tloss / max(1, tcount)), (tcorrect / max(1, tcount))

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs,
               writer, run_tag, save_all=False, verbose=False, use_mixup=False, mixup_alpha=0.2, use_tta=False,
               # NEW:
               run_name=None, stage="head", start_epoch=0, backbone="efficientnet_b0", num_classes=7):
    best_val_acc = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PREC)
    # e is zero-based absolute index within the stage
    for e in range(start_epoch, start_epoch + epochs):
        epoch = e - start_epoch + 1  # 1..epochs for display

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha
        )
        if scheduler: scheduler.step()
        val_loss, val_acc, *_ = evaluate(model, val_loader, DEVICE, use_tta=use_tta)

        writer.add_scalars(f"{run_tag}/Loss", {"train":train_loss, "val":val_loss}, epoch)
        writer.add_scalars(f"{run_tag}/Acc", {"train":train_acc, "val":val_acc}, epoch)
        print(f"[{run_tag}] Epoch {epoch:02d}/{epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if save_all:
            torch.save(model.state_dict(), CKPT_DIR/f"{run_tag}_epoch{epoch:02d}.pt")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR/f"{run_name}_best.pt")

        # NEW: always save a resumable last checkpoint for this stage
        save_last_ckpt(
            CKPT_DIR/f"{run_name}_last.pt",
            e, model, optimizer, scheduler, run_name, stage, backbone, num_classes
        )
    return best_val_acc

def main():
    seed_everything(42)
    args = get_args()

    print(f"[env] device={DEVICE}, mixed_precision={MIXED_PREC}")
    print(f"[cfg] backbone={args.backbone} bs={args.batch_size} lr_head={args.lr_head} lr_ft={args.lr_ft}")

    # Data
    label_to_index = load_class_map()
    num_classes = len(label_to_index)
    labels = [k for k,_ in sorted(label_to_index.items(), key=lambda x: x[1])]
    print(f"[data] classes={labels}")
    print("[data] building loaders…")
    train_loader, val_loader, test_loader = make_loaders(batch_size=args.batch_size)
    print(f"[data] train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader) if test_loader else 0}")
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("Empty train/val loader. Check your data paths and class folders.")

    # Model
    print("[model] building backbone…")
    model, last_prefixes = build_model(args.backbone, num_classes)
    total_p, train_p = count_params(model)
    print(f"[model] params total={total_p:,} trainable_now={train_p:,}")
    model = model.to(DEVICE)

    # Losses
    try:
        weights = class_weights_from_audit(label_to_index).to(DEVICE)
        print("[loss] using class weights from audit.")
    except Exception as e:
        print("[loss] WARN could not read audit weights, defaulting to uniform. Error:", e)
        weights = None

    # Stage-specific criteria:
    criterion_head = nn.CrossEntropyLoss(weight=weights)                        # no smoothing (head/partial unfreeze)
    criterion_ft   = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)  # smoothing for fine-tune

    # Run setup
    run_name = f"{args.backbone}_tl_{int(time.time())}"
    run_dir = OUT_DIR/"runs"/run_name
    writer = SummaryWriter(str(run_dir))
    (OUT_DIR/"metrics").mkdir(parents=True, exist_ok=True)
    (run_dir/"manifest.txt").write_text(
        f"backbone={args.backbone}\n"
        f"epochs_head={args.epochs_head}\nepochs_ft={args.epochs_ft}\n"
        f"lr_head={args.lr_head}\nlr_ft={args.lr_ft}\n"
        f"batch_size={args.batch_size}\npatience={args.patience}\n"
        f"use_mixup={args.use_mixup} alpha={args.mixup_alpha}\nuse_tta={args.use_tta}\n"
        f"labels={labels}\n"
    )
    print(f"[run] logdir={run_dir}")

    # ---- Resume pre-read (we'll load states after creating opt/sched) ----
    resume_info = None
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        print(f"[resume] Loading {args.resume} (stage={ckpt.get('stage')}, epoch={ckpt.get('epoch')})")
        try:
            model.load_state_dict(ckpt["model_state"])
        except Exception as e:
            print("[resume] FAILED to load model_state:", e); raise
        resume_info = {"epoch": int(ckpt.get("epoch", 0)),
                       "stage": ckpt.get("stage", "ft")}

    # -------- Stage 1: PARTIAL UNFREEZE + head (no LS, no MixUp, no TTA) --------
    print("[stage1] partial unfreeze (last blocks) + head warm-up…")
    set_trainable(model, False)
    unfreeze_selected(model, last_prefixes)          # <-- allow last blocks to adapt in warm-up
    unfreeze_selected(model, ["fc","classifier"])    # ensure head is trainable
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr_head, weight_decay=args.weight_decay)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs_head)

    start_epoch_head = 0
    if resume_info and resume_info["stage"] == "head":
        print(f"[resume] Resuming Stage-1 (head) from epoch {resume_info['epoch']+1}")
        try:
            if ckpt.get("optimizer_state"): opt1.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") and sched1: sched1.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            print("[resume] WARN could not load head optimizer/scheduler state:", e)
        start_epoch_head = resume_info["epoch"] + 1

    if args.epochs_head > 0 and not (resume_info and resume_info["stage"] == "ft"):
        best_val_head = train_loop(
            model, train_loader, val_loader, criterion_head, opt1, sched1,
            args.epochs_head, writer, run_tag=run_name+"_head",
            save_all=args.save_all_epochs, verbose=args.debug,
            use_mixup=False, mixup_alpha=0.0, use_tta=False,
            # NEW
            run_name=run_name, stage="head", start_epoch=start_epoch_head,
            backbone=args.backbone, num_classes=num_classes
        )
        print(f"[stage1] best_val_acc={best_val_head:.4f}")
        torch.save(model.state_dict(), CKPT_DIR/f"{run_name}_after_head.pt")
    else:
        print("[stage1] skipped (epochs_head=0 or resuming into FT)")

    # -------- Stage 2: Unfreeze selected blocks (again) & fine-tune --------
    print("[stage2] unfreezing selected blocks, fine-tuning…")
    set_trainable(model, False)
    unfreeze_selected(model, last_prefixes)          # make sure those blocks + head are trainable
    unfreeze_selected(model, ["fc","classifier"])
    total_p2, train_p2 = count_params(model)
    print(f"[stage2] trainable_params_now={train_p2:,}")

    opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr_ft, weight_decay=args.weight_decay)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs_ft)

    best_so_far = 0.0
    epochs_no_improve = 0
    scaler = torch.amp.GradScaler("cuda", enabled=MIXED_PREC)

    start_epoch_ft = 0
    if resume_info and resume_info["stage"] == "ft":
        print(f"[resume] Resuming Stage-2 (ft) from epoch {resume_info['epoch']+1}")
        try:
            if ckpt.get("optimizer_state"): opt2.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") and sched2: sched2.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            print("[resume] WARN could not load ft optimizer/scheduler state:", e)
        start_epoch_ft = resume_info["epoch"] + 1

    for e in range(start_epoch_ft, start_epoch_ft + args.epochs_ft):
        epoch = e - start_epoch_ft + 1
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_ft, opt2, scaler,
            use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha
        )
        sched2.step()
        val_loss, val_acc, *_ = evaluate(model, val_loader, DEVICE, use_tta=args.use_tta)

        writer.add_scalars(f"{run_name}/FT_Loss", {"train":train_loss, "val":val_loss}, epoch)
        writer.add_scalars(f"{run_name}/FT_Acc", {"train":train_acc, "val":val_acc}, epoch)
        print(f"[{run_name}/ft] Epoch {epoch:02d}/{args.epochs_ft} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if args.save_all_epochs:
            torch.save(model.state_dict(), CKPT_DIR/f"{run_name}_ft_epoch{epoch:02d}.pt")

        if val_acc > best_so_far + 1e-4:
            best_so_far = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR/f"{run_name}_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("[stage2] Early stopping triggered.")
                # still save a last checkpoint before breaking
                save_last_ckpt(
                    CKPT_DIR/f"{run_name}_last.pt",
                    e, model, opt2, sched2, run_name, stage="ft",
                    backbone=args.backbone, num_classes=num_classes
                )
                break

        # NEW: save resumable last checkpoint after each FT epoch
        save_last_ckpt(
            CKPT_DIR/f"{run_name}_last.pt",
            e, model, opt2, sched2, run_name, stage="ft",
            backbone=args.backbone, num_classes=num_classes
        )

    # ------- Final eval (best ckpt) -------
    best_path = CKPT_DIR/f"{run_name}_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_path:=DEVICE))

    rep_dir = OUT_DIR/"metrics"; rep_dir.mkdir(parents=True, exist_ok=True)
    _, val_acc, yv, pv = evaluate(model, val_loader, DEVICE, use_tta=args.use_tta)
    (rep_dir/f"{run_name}_val_report.txt").write_text(
        classification_report(yv, pv, target_names=labels, digits=4, zero_division=0))
    np.savetxt(rep_dir/f"{run_name}_val_confusion_matrix.csv",
               confusion_matrix(yv, pv), fmt="%d", delimiter=",")
    print(f"[final] Val accuracy (best ckpt): {val_acc:.4f}")

    if test_loader is not None and len(test_loader.dataset) > 0:
        _, test_acc, yt, pt = evaluate(model, test_loader, DEVICE, use_tta=args.use_tta)
        (rep_dir/f"{run_name}_test_report.txt").write_text(
            classification_report(yt, pt, target_names=labels, digits=4, zero_division=0))
        np.savetxt(rep_dir/f"{run_name}_test_confusion_matrix.csv",
                   confusion_matrix(yt, pt), fmt="%d", delimiter=",")
        print(f"[final] Test accuracy: {test_acc:.4f}")

    print("[done] Reports saved to:", rep_dir)

if __name__ == "__main__":
    print("[train_transfer] starting…")
    try:
        main()
    except Exception as e:
        err_path = OUT_DIR/"logs"/"train_transfer_error.log"
        with open(err_path, "w") as f:
            f.write("ERROR during train_transfer.py\n")
            f.write("".join(traceback.format_exception(*sys.exc_info())))
        print(f"[FATAL] An error occurred. See log: {err_path}")
        raise
