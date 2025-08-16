import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# --- Optional deps (lazy import to keep import cost low) ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

# onnxruntime is optional
try:
    import onnxruntime as ort
    _ORT_OK = True
except Exception:
    _ORT_OK = False


# --------------------- Paths & labels ---------------------
DATA_DIR = Path("data")
CKPT_DIR = Path("models/checkpoints")
FINAL_DIR = Path("models/final_model")

def get_labels() -> List[str]:
    """Return class labels in index order."""
    with open(DATA_DIR/"metadata"/"classes.json") as f:
        m = json.load(f)["label_to_index"]
    return [k for k,_ in sorted(m.items(), key=lambda kv: kv[1])]


# --------------------- Utils ---------------------
def _latest_best() -> Optional[Path]:
    cks = sorted(CKPT_DIR.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cks[0] if cks else None

def _infer_backbone_from_name(name: str) -> str:
    name = name.lower()
    if "efficientnet_b0" in name: return "efficientnet_b0"
    if "resnet50" in name: return "resnet50"
    # default to your main backbone
    return "efficientnet_b0"

def _build_torch_model(backbone: str, num_classes: int) -> "nn.Module":
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available; install torch/torchvision to use engine='pytorch' or 'torchscript'.")
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


# --------------------- Preprocessing ---------------------
def _resize_side(img_size: int) -> int:
    # standard 224 -> resize short side ≈256
    return 256 if img_size == 224 else int(round(img_size * 1.14))

def _build_transform(img_size: int):
    if not _TORCH_OK:
        raise RuntimeError("PyTorch/torchvision required for preprocessing.")
    return transforms.Compose([
        transforms.Resize(_resize_side(img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def preprocess_pil(img: Image.Image, img_size: int = 224) -> "torch.Tensor":
    """Return a 4D tensor (1,C,H,W) suitable for model input."""
    tfm = _build_transform(img_size)
    x = tfm(img.convert("RGB")).unsqueeze(0)
    return x


# --------------------- Inference wrapper ---------------------
class DeepFERModel:
    """
    A thin wrapper that supports engines:
      - 'pytorch': eager PyTorch
      - 'torchscript': TorchScript .pt
      - 'onnx': ONNX Runtime .onnx
    """
    def __init__(self,
                 engine: str = "pytorch",
                 ckpt: Optional[Union[str, Path]] = None,
                 img_size: int = 224,
                 use_tta: bool = False,
                 device: Optional[str] = None):
        self.engine = engine.lower()
        self.img_size = img_size
        self.use_tta = use_tta
        self.labels = get_labels()
        self.k = len(self.labels)

        # resolve checkpoint/export
        ckpt_path: Optional[Path] = None
        if ckpt:
            ckpt_path = Path(ckpt)
        else:
            # prefer final_model artifacts if present
            if self.engine == "onnx" and (FINAL_DIR/"model.onnx").exists():
                ckpt_path = FINAL_DIR/"model.onnx"
            elif self.engine == "torchscript" and (FINAL_DIR/"model_ts.pt").exists():
                ckpt_path = FINAL_DIR/"model_ts.pt"
            elif self.engine == "pytorch" and (FINAL_DIR/"model_best.pt").exists():
                ckpt_path = FINAL_DIR/"model_best.pt"
            # else fall back to latest *_best.pt
            if ckpt_path is None and self.engine in {"pytorch", "torchscript"}:
                ckpt_path = _latest_best()

        if self.engine == "pytorch":
            if not _TORCH_OK: raise RuntimeError("PyTorch not available.")
            assert ckpt_path is not None, "No *_best.pt found; pass --ckpt"
            run_tag = ckpt_path.stem.replace("_best","")
            backbone = _infer_backbone_from_name(run_tag)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = _build_torch_model(backbone, self.k).to(self.device)
            state = torch.load(ckpt_path, map_location=self.device)
            # If your torch warns about weights_only in future, you can set weights_only=True
            self.model.load_state_dict(state)
            self.model.eval()
        elif self.engine == "torchscript":
            if not _TORCH_OK: raise RuntimeError("PyTorch not available.")
            assert ckpt_path is not None and ckpt_path.suffix == ".pt", "Provide a TorchScript .pt file"
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.jit.load(str(ckpt_path), map_location=self.device).eval()
        elif self.engine == "onnx":
            if not _ORT_OK: raise RuntimeError("onnxruntime is not installed. pip install onnxruntime-gpu or onnxruntime")
            assert ckpt_path is not None and ckpt_path.suffix == ".onnx", "Provide an ONNX .onnx file"
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                self.sess = ort.InferenceSession(str(ckpt_path), providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            else:
                self.sess = ort.InferenceSession(str(ckpt_path), providers=["CPUExecutionProvider"])
            self.in_name = self.sess.get_inputs()[0].name
            self.device = "cpu"  # ORT takes numpy on CPU memory
        else:
            raise ValueError(f"Unknown engine: {engine}")

    # ---------- core predict ----------
    def _softmax_pt(self, logits: "torch.Tensor") -> np.ndarray:
        return F.softmax(logits, dim=1).detach().cpu().numpy()

    def _run_pt(self, x: "torch.Tensor") -> np.ndarray:
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            if self.use_tta:
                logits = (logits + self.model(torch.flip(x.to(self.device), dims=[-1]))) / 2.0
        return self._softmax_pt(logits)

    def _run_ort(self, x: "torch.Tensor") -> np.ndarray:
        x_np = x.numpy().astype(np.float32)
        if self.use_tta:
            x_flip = torch.flip(x, dims=[-1]).numpy().astype(np.float32)
            p1 = self.sess.run(None, {self.in_name: x_np})[0]
            p2 = self.sess.run(None, {self.in_name: x_flip})[0]
            probs = (p1 + p2) / 2.0
        else:
            probs = self.sess.run(None, {self.in_name: x_np})[0]
        # Ensure softmax if model outputs logits (export kept logits)
        exps = np.exp(probs - probs.max(axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        return probs

    # ---------- public API ----------
    def predict_tensor(self, x4d: "torch.Tensor") -> np.ndarray:
        """x4d must be (N,3,H,W), normalized."""
        if self.engine in {"pytorch", "torchscript"}:
            return self._run_pt(x4d)
        else:
            return self._run_ort(x4d)

    def predict_pil(self, img: Image.Image, topk: int = 3) -> Dict:
        x = preprocess_pil(img, img_size=self.img_size)
        probs = self.predict_tensor(x)[0]
        idx = int(np.argmax(probs))
        order = np.argsort(-probs)[:topk]
        return {
            "top1": {"label": self.labels[idx], "prob": float(probs[idx])},
            "topk": [
                {"label": self.labels[i], "prob": float(probs[i])} for i in order
            ],
            "probs": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        }

    def predict_file(self, path: Union[str, Path], topk: int = 3) -> Dict:
        img = Image.open(path).convert("RGB")
        out = self.predict_pil(img, topk=topk)
        out["image"] = str(path)
        return out


# --------------------- Public helpers ---------------------
def load_model(engine: str = "pytorch",
               ckpt: Optional[Union[str, Path]] = None,
               img_size: int = 224,
               use_tta: bool = False,
               device: Optional[str] = None) -> DeepFERModel:
    """
    Load a DeepFERModel.
    engine: 'pytorch' | 'torchscript' | 'onnx'
    ckpt: path to model_best.pt / model_ts.pt / model.onnx (optional; auto-resolves)
    """
    return DeepFERModel(engine=engine, ckpt=ckpt, img_size=img_size, use_tta=use_tta, device=device)

def predict(image_path: Union[str, Path],
            engine: str = "pytorch",
            ckpt: Optional[Union[str, Path]] = None,
            img_size: int = 224,
            use_tta: bool = False,
            topk: int = 3) -> Dict:
    """One-shot prediction from image file path."""
    model = load_model(engine=engine, ckpt=ckpt, img_size=img_size, use_tta=use_tta)
    return model.predict_file(image_path, topk=topk)

def predict_batch(image_paths: List[Union[str, Path]],
                  engine: str = "pytorch",
                  ckpt: Optional[Union[str, Path]] = None,
                  img_size: int = 224,
                  use_tta: bool = False,
                  topk: int = 3) -> List[Dict]:
    """Batch convenience wrapper (loops predict_file)."""
    model = load_model(engine=engine, ckpt=ckpt, img_size=img_size, use_tta=use_tta)
    return [model.predict_file(p, topk=topk) for p in image_paths]


# --------------------- CLI ---------------------
def _cli():
    import argparse, json as _json
    ap = argparse.ArgumentParser(description="DeepFER inference CLI")
    ap.add_argument("--image", type=str, help="Path to an image")
    ap.add_argument("--dir", type=str, help="Folder with images (jpg/png/jpeg)")
    ap.add_argument("--engine", type=str, default="pytorch", choices=["pytorch","torchscript","onnx"])
    ap.add_argument("--ckpt", type=str, default=None, help="Optional: path to model file")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--use_tta", action="store_true")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--save_json", type=str, default=None, help="Optional output JSON path")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("Provide --image or --dir")

    model = load_model(engine=args.engine, ckpt=args.ckpt, img_size=args.img_size, use_tta=args.use_tta)

    results = []
    if args.image:
        results = [ model.predict_file(args.image, topk=args.topk) ]
    else:
        exts = {".jpg",".jpeg",".png",".bmp",".webp"}
        paths = [p for p in Path(args.dir).rglob("*") if p.suffix.lower() in exts]
        for p in sorted(paths):
            results.append(model.predict_file(p, topk=args.topk))

    # pretty print
    print(_json.dumps(results if len(results)>1 else results[0], indent=2))

    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(_json.dumps(results, indent=2))

if __name__ == "__main__":
    _cli()
