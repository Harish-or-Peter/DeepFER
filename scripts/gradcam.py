import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets

# ------------------- Paths & Device -------------------
DATA_DIR = Path("data")
CKPT_DIR = Path("models/checkpoints")
OUT_DIR  = Path("results/predictions/gradcam"); OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- CLI -------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="Path to *_best.pt (default: latest)")
    ap.add_argument("--split", type=str, default="val", choices=["val","test"], help="Dataset split to sample from")
    ap.add_argument("--per_class", type=int, default=6, help="How many images per class to visualize (ignored if --images given)")
    ap.add_argument("--images", nargs="*", default=None, help="Optional list of image file paths")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--top_only", action="store_true", help="Only generate CAM for predicted class (skip true label CAM)")
    ap.add_argument("--grid", action="store_true", help="Also save a side-by-side grid (Original | Pred-CAM | True-CAM)")
    return ap.parse_args()

# ------------------- Model Builders -------------------
def infer_backbone_from_name(name: str):
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
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m

def load_labels():
    with open(DATA_DIR/"metadata"/"classes.json") as f:
        m = json.load(f)["label_to_index"]
    return [k for k,_ in sorted(m.items(), key=lambda kv: kv[1])]

def latest_best_ckpt():
    cands = sorted(CKPT_DIR.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

# ------------------- Grad-CAM Core -------------------
class GradCAM:
    """
    Generic Grad-CAM that auto-picks the last 4D feature map module as target.
    """
    def __init__(self, model: nn.Module, input_size=(1,3,224,224)):
        self.model = model
        self.model.eval()
        self.target_module = self._auto_pick_target_layer(input_size)
        self.activations = None
        self.gradients = None
        # register hooks on target_module
        def fwd_hook(m, inp, out):
            self.activations = out.detach()  # N,C,H,W
        def bwd_hook(m, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_module.register_forward_hook(fwd_hook)
        self.target_module.register_backward_hook(bwd_hook)

    def _auto_pick_target_layer(self, input_size):
        feats = []
        hooks = []
        def save_4d(mod, inp, out):
            if torch.is_tensor(out) and out.dim() == 4:
                feats.append((mod, out.shape))
        for m in self.model.modules():
            hooks.append(m.register_forward_hook(lambda m,i,o: save_4d(m,i,o)))
        with torch.no_grad():
            _ = self.model(torch.zeros(*input_size, device=DEVICE))
        for h in hooks: h.remove()
        assert feats, "No 4D activation found; model not supported."
        return feats[-1][0]

    def __call__(self, x: torch.Tensor, class_idx: int):
        """
        x: (1,3,H,W) normalized tensor on DEVICE
        class_idx: target class for which to compute CAM
        returns: (H,W) np.float32 heatmap in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        score = logits[:, class_idx].sum()
        score.backward()

        A = self.activations          # (1,C,h,w)
        dA = self.gradients           # (1,C,h,w)
        assert A is not None and dA is not None, "Hooks did not capture activations/gradients."
        weights = dA.mean(dim=(2,3), keepdim=True)  # (1,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=False)  # (1,h,w)
        cam = F.relu(cam)[0]                            # (h,w)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.detach().cpu().numpy().astype(np.float32)

# ------------------- Transforms & Overlay -------------------
def build_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(256 if img_size==224 else int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def build_display_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(256 if img_size==224 else int(img_size*1.14)),
        transforms.CenterCrop(img_size),
    ])

def overlay_cam_on_image(pil_img, cam, alpha=0.35):
    # cam: (H,W) in [0,1]
    cam_img = (cam * 255).astype(np.uint8)
    cmap = plt.get_cmap("jet")
    colored = (cmap(cam_img/255.0)[:, :, :3] * 255).astype(np.uint8)  # RGB
    colored = Image.fromarray(colored).resize(pil_img.size, resample=Image.BILINEAR)
    blended = Image.blend(pil_img.convert("RGB"), colored, alpha=alpha)
    return blended

# ------------------- Grid helpers -------------------
def add_caption(img: Image.Image, text: str, bar_height: int = 26):
    img = img.copy()
    bar = Image.new("RGBA", (img.width, bar_height), (0,0,0,170))
    img.paste(bar, (0,0), bar)
    d = ImageDraw.Draw(img)
    d.text((6,4), text, fill=(255,255,255))
    return img.convert("RGB")

def hstack_images(imgs, pad=8, bg=(255,255,255)):
    imgs = [im.convert("RGB") for im in imgs if im is not None]
    if not imgs: return None
    h = max(im.height for im in imgs)
    widths = sum(im.width for im in imgs) + pad*(len(imgs)-1)
    canvas = Image.new("RGB", (widths, h), bg)
    x = 0
    for im in imgs:
        if im.height != h:
            im = im.resize((im.width, h), Image.BILINEAR)
        canvas.paste(im, (x, 0))
        x += im.width + pad
    return canvas

# ------------------- Dataset helpers -------------------
def sample_paths_from_split(split_root: Path, per_class: int):
    ds = datasets.ImageFolder(split_root)
    buckets = {i: [] for i in range(len(ds.classes))}
    for p, y in ds.samples:
        buckets[y].append(p)
    import random
    out = []
    for y, lst in buckets.items():
        random.seed(42)
        random.shuffle(lst)
        out.extend(lst[:per_class])
    return out, [Path(p).parent.name for p in out]

# ------------------- Main -------------------
def main():
    args = get_args()
    labels = load_labels()
    num_classes = len(labels)

    ckpt_path = Path(args.ckpt) if args.ckpt else latest_best_ckpt()
    assert ckpt_path is not None, "No *_best.pt found; specify --ckpt"
    run_tag = ckpt_path.stem.replace("_best","")
    backbone = infer_backbone_from_name(run_tag)
    print(f"[gradcam] using checkpoint: {ckpt_path.name} | backbone={backbone}")

    model = build_model(backbone, num_classes).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    cam_engine = GradCAM(model, input_size=(1,3,args.img_size,args.img_size))
    t_eval = build_val_transform(args.img_size)
    t_disp = build_display_transform(args.img_size)

    if args.images:
        img_paths = [Path(p) for p in args.images]
        true_names = [p.parent.name for p in img_paths]
    else:
        split_dir = DATA_DIR / args.split
        img_paths, true_names = sample_paths_from_split(split_dir, args.per_class)

    outdir = OUT_DIR / run_tag
    outdir.mkdir(parents=True, exist_ok=True)

    for i, (p, tname) in enumerate(zip(img_paths, true_names), 1):
        try:
            pil_disp = t_disp(Image.open(p).convert("RGB"))
            x = t_eval(pil_disp).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(probs.argmax())
                pred_name = labels[pred_idx]
                pred_prob = float(probs[pred_idx])

            # Pred CAM
            torch.set_grad_enabled(True)
            x.requires_grad_(True)
            cam_pred = cam_engine(x, class_idx=pred_idx)
            pred_overlay = overlay_cam_on_image(pil_disp, cam_pred, alpha=0.40)

            # True CAM (optional)
            save_true = (not args.top_only) and (tname in labels)
            true_overlay = None
            if save_true:
                true_idx = labels.index(tname)
                cam_true = cam_engine(x, class_idx=true_idx)
                true_overlay = overlay_cam_on_image(pil_disp, cam_true, alpha=0.40)

            stem = Path(p).stem
            # Save individual overlays (as before)
            pred_out = outdir / f"{stem}__true-{tname}__pred-{pred_name}__p{pred_prob:.2f}__PRED.png"
            pred_overlay.save(pred_out)
            if save_true:
                true_out = outdir / f"{stem}__true-{tname}__pred-{pred_name}__p{pred_prob:.2f}__TRUE.png"
                true_overlay.save(true_out)

            # Save grid if requested
            if args.grid:
                cap_orig = add_caption(pil_disp, f"ORIG • true:{tname}")
                cap_pred = add_caption(pred_overlay, f"PRED:{pred_name} ({pred_prob:.2f})")
                cap_true = add_caption(true_overlay, f"TRUE:{tname}") if save_true else None
                grid = hstack_images([cap_orig, cap_pred, cap_true], pad=10)
                grid_out = outdir / f"{stem}__true-{tname}__pred-{pred_name}__p{pred_prob:.2f}__GRID.png"
                if grid is not None:
                    grid.save(grid_out)

            print(f"[{i:03d}/{len(img_paths)}] {p.name}: pred={pred_name}({pred_prob:.2f}) true={tname} -> saved")

        except Exception as e:
            print(f"[warn] failed on {p}: {e}")

    print(f"[done] Grad-CAM images saved at: {outdir}")

if __name__ == "__main__":
    print(f"[gradcam] device={DEVICE}")
    main()
