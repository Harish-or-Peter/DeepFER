# scripts/serve/api.py
from __future__ import annotations
import io, os, json, time, base64, glob
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import onnxruntime as ort

# ---------- Paths & constants ----------
ROOT = Path(__file__).resolve().parents[2]  # .../DeepFER_Project
DATA_DIR = ROOT / "data"
META_FILE = DATA_DIR / "metadata" / "classes.json"

MODELS_DIR = ROOT / "models"
FINAL_DIR = MODELS_DIR / "final_model"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORT_DIR = ROOT / "scripts" / "export"  # optional: where you saved ONNX/TS

PT_BEST = (FINAL_DIR / "model_best.pt") if (FINAL_DIR / "model_best.pt").exists() else None
TS_FILE = (FINAL_DIR / "model.ts") if (FINAL_DIR / "model.ts").exists() else (EXPORT_DIR / "model.ts")
ONNX_FILE = (FINAL_DIR / "model.onnx") if (FINAL_DIR / "model.onnx").exists() else (EXPORT_DIR / "model.onnx")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# exactly as in training
PREPROC = T.Compose([
    T.Resize(256, interpolation=InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

DARK_MEAN_THRESH = 0.15           # 0..1 mean brightness; treat as "unknown"
MIN_FACE_SIDE = 40                # px (reject tiny crops)
MIN_FACE_SCORE_DEFAULT = 0.60     # MediaPipe confidence gate

# ---------- Utilities ----------
def load_labels() -> List[str]:
    with open(META_FILE, "r", encoding="utf-8") as f:
        m = json.load(f)
    # labels are 0..N-1
    return [k for k, _ in sorted(m["label_to_index"].items(), key=lambda kv: kv[1])]

LABELS: List[str] = load_labels()
NUM_CLASSES = len(LABELS)

def build_model(backbone: str = "efficientnet_b0", num_classes: int = NUM_CLASSES) -> nn.Module:
    from torchvision import models
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        # dropout head (same as training script)
        if hasattr(model.classifier[0], "p"):
            model.classifier[0] = nn.Dropout(p=0.4)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model

def infer_backbone_from_manifest() -> str:
    manifest = FINAL_DIR / "manifest.txt"
    if manifest.exists():
        txt = manifest.read_text()
        for line in txt.splitlines():
            if line.startswith("backbone="):
                return line.split("=", 1)[1].strip()
    return "efficientnet_b0"

BACKBONE = infer_backbone_from_manifest()

def find_latest_best_ckpt() -> Optional[Path]:
    pats = sorted(CHECKPOINTS_DIR.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pats[0] if pats else None

# ---- Face detection (MediaPipe) ----
try:
    import mediapipe as mp
    _mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=MIN_FACE_SCORE_DEFAULT
    )
except Exception:
    _mp_fd = None

def crop_face_np(img_rgb: np.ndarray, margin: float = 0.25, min_face_score: float = MIN_FACE_SCORE_DEFAULT):
    """
    img_rgb: HxWx3 uint8 RGB; returns (face_rgb_np, score) or (None, score_or_None)
    """
    if _mp_fd is None:
        return None, None
    res = _mp_fd.process(img_rgb)  # MediaPipe wants RGB numpy
    if not res.detections:
        return None, None
    det = max(res.detections, key=lambda d: float(d.score[0]))
    score = float(det.score[0])
    if score < min_face_score:
        return None, score
    bbox = det.location_data.relative_bounding_box
    h, w, _ = img_rgb.shape
    x0 = int((bbox.xmin - margin) * w)
    y0 = int((bbox.ymin - margin) * h)
    x1 = int((bbox.xmin + bbox.width + margin) * w)
    y1 = int((bbox.ymin + bbox.height + margin) * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if (x1 - x0) < MIN_FACE_SIDE or (y1 - y0) < MIN_FACE_SIDE:
        return None, score
    face = img_rgb[y0:y1, x0:x1]
    return face, score

def preprocess_rgb_np(img_rgb: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(img_rgb)  # keep RGB
    ten = PREPROC(pil).unsqueeze(0).to(DEVICE)
    return ten

# ---------- Backends ----------
_PYTORCH_MODEL: Optional[nn.Module] = None
_TORCHSCRIPT: Optional[torch.jit.ScriptModule] = None
_ONNX: Optional[ort.InferenceSession] = None
_ONNX_IN_NAME = None
_ONNX_OUT_NAME = None

def load_pytorch_model():
    global _PYTORCH_MODEL
    if _PYTORCH_MODEL is not None:
        return _PYTORCH_MODEL
    ckpt = PT_BEST if PT_BEST else find_latest_best_ckpt()
    if ckpt is None:
        raise FileNotFoundError("No PyTorch checkpoint found (expected models/final_model/model_best.pt).")
    model = build_model(BACKBONE, NUM_CLASSES)
    sd = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    _PYTORCH_MODEL = model
    return _PYTORCH_MODEL

def load_torchscript():
    global _TORCHSCRIPT
    if _TORCHSCRIPT is not None:
        return _TORCHSCRIPT
    if not TS_FILE.exists():
        # fall back to scripting current PyTorch model
        model = load_pytorch_model()
        _TORCHSCRIPT = torch.jit.script(model)
    else:
        _TORCHSCRIPT = torch.jit.load(str(TS_FILE), map_location=DEVICE)
    _TORCHSCRIPT.eval().to(DEVICE)
    return _TORCHSCRIPT

def load_onnx():
    global _ONNX, _ONNX_IN_NAME, _ONNX_OUT_NAME
    if _ONNX is not None:
        return _ONNX
    if not ONNX_FILE.exists():
        raise FileNotFoundError("ONNX model not found (expected models/final_model/model.onnx).")
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _ONNX = ort.InferenceSession(str(ONNX_FILE), providers=providers)
    inps = _ONNX.get_inputs()
    outs = _ONNX.get_outputs()
    _ONNX_IN_NAME = inps[0].name
    _ONNX_OUT_NAME = outs[0].name
    return _ONNX

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def run_model_and_get_probs(x: torch.Tensor, engine: str = "onnx", use_tta: bool = False) -> np.ndarray:
    """
    x: 1x3x224x224 torch tensor (DEVICE)
    returns numpy probs [C]
    """
    engine = (engine or "onnx").lower()
    outs = []

    def _torch_forward(mdl, xt):
        with torch.no_grad():
            logits = mdl(xt)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            return probs

    if engine == "pytorch":
        mdl = load_pytorch_model()
        outs.append(_torch_forward(mdl, x))
        if use_tta:
            outs.append(_torch_forward(mdl, torch.flip(x, dims=[3])))
        return np.mean(outs, axis=0)

    elif engine == "torchscript":
        mdl = load_torchscript()
        outs.append(_torch_forward(mdl, x))
        if use_tta:
            outs.append(_torch_forward(mdl, torch.flip(x, dims=[3])))
        return np.mean(outs, axis=0)

    else:  # onnx
        sess = load_onnx()
        import onnxruntime as ort  # noqa
        def _onnx_forward(xt):
            arr = xt.detach().cpu().numpy().astype("float32")
            pred = sess.run([_ONNX_OUT_NAME], {_ONNX_IN_NAME: arr})[0][0]  # logits or probs depending on export
            # be safe: convert to probs if not already
            if pred.ndim == 1 and pred.max() <= 1.0 and pred.min() >= 0.0 and abs(pred.sum() - 1.0) < 1e-3:
                return pred.astype(float)
            return softmax_np(pred.astype(float))
        outs.append(_onnx_forward(x))
        if use_tta:
            outs.append(_onnx_forward(torch.flip(x, dims=[3])))
        return np.mean(outs, axis=0)

# ---------- FastAPI ----------
app = FastAPI(title="DeepFER API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class PredictB64Request(BaseModel):
    image_b64: str
    engine: Optional[str] = "onnx"      # 'onnx' | 'torchscript' | 'pytorch'
    use_tta: Optional[bool] = False
    detect_face: Optional[bool] = True
    unknown_thresh: Optional[float] = 0.5
    min_face_score: Optional[float] = MIN_FACE_SCORE_DEFAULT
    topk: Optional[int] = 5

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "labels": LABELS,
        "backbone": BACKBONE,
        "pt_best": str(PT_BEST) if PT_BEST else None,
        "onnx": str(ONNX_FILE) if ONNX_FILE.exists() else None,
        "ts": str(TS_FILE) if TS_FILE.exists() else None
    }

@app.get("/labels")
def labels():
    return {"labels": LABELS}

@app.post("/predict-b64")
def predict_b64(req: PredictB64Request):
    # Decode base64 → PIL → numpy RGB
    try:
        b = base64.b64decode(req.image_b64.split(",")[-1])
        pil = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return {"top1": {"label": "unknown", "prob": 1.0}, "unknown": True, "reason": "decode_error", "topk": []}

    np_rgb = np.array(pil)  # HWC uint8

    # Gate: too dark?
    if (np_rgb.mean() / 255.0) < DARK_MEAN_THRESH:
        return {
            "top1": {"label": "unknown", "prob": 1.0},
            "unknown": True,
            "reason": "too_dark",
            "topk": [],
            "face_score": None,
        }

    # Face detect → crop
    face_score = None
    if req.detect_face:
        face, face_score = crop_face_np(
            np_rgb, margin=0.25,
            min_face_score=float(req.min_face_score or MIN_FACE_SCORE_DEFAULT)
        )
        if face is None:
            return {
                "top1": {"label": "unknown", "prob": 1.0},
                "unknown": True,
                "reason": "no_or_lowconf_face",
                "topk": [],
                "face_score": face_score
            }
        np_rgb = face

    # Preprocess
    x = preprocess_rgb_np(np_rgb)

    # Run model
    probs = run_model_and_get_probs(x, engine=req.engine or "onnx", use_tta=bool(req.use_tta))
    pmax_idx = int(probs.argmax())
    pmax = float(probs[pmax_idx])
    pred_label = LABELS[pmax_idx]

    # Unknown thresholding
    unk_th = float(req.unknown_thresh or 0.5)
    is_unknown = (pmax < unk_th)

    k = int(req.topk or 5)
    topk_idx = probs.argsort()[-k:][::-1]
    topk = [{"label": LABELS[j], "prob": float(probs[j])} for j in topk_idx]

    return {
        "top1": {"label": ("unknown" if is_unknown else pred_label), "prob": pmax},
        "unknown": is_unknown,
        "face_score": face_score,
        "topk": topk
    }
