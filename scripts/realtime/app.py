import argparse, time, csv, json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# -------- Paths & device --------
DATA_DIR = Path("data")
CKPT_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"; METRICS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = RESULTS_DIR / "predictions"; PRED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0 for webcam, or path to video")
    ap.add_argument("--backend", type=str, default="mediapipe", choices=["mediapipe","opencv"])
    ap.add_argument("--ckpt", type=str, default=None, help="Path to *_best.pt (defaults to latest)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--margin", type=float, default=0.18, help="crop margin fraction")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=str(PRED_DIR / "demo_emotions.mp4"))
    ap.add_argument("--use_tta", action="store_true", help="hflip TTA at inference")
    ap.add_argument("--ema_alpha", type=float, default=0.25, help="EMA smoothing for probs [0..1]")
    ap.add_argument("--unknown_thresh", type=float, default=0.45, help="label 'unknown' if max prob < thresh")
    ap.add_argument("--max_frames", type=int, default=0)
    return ap.parse_args()

# -------- helpers --------
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

def clamp(x1, y1, x2, y2, w, h):
    return max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)

def add_margin(x1, y1, x2, y2, w, h, mfrac):
    bw, bh = x2 - x1, y2 - y1
    m = int(max(bw, bh) * mfrac)
    return clamp(x1 - m, y1 - m, x2 + m, y2 + m, w, h)

def preprocess_crop(bgr, img_size):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    tfm = transforms.Compose([
        transforms.Resize(256 if img_size==224 else int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(pil).unsqueeze(0)

def softmax_infer(model, x, use_tta=False):
    with torch.no_grad():
        logits = model(x)
        if use_tta:
            # hflip TTA
            x_flip = torch.flip(x, dims=[-1])
            logits = (logits + model(x_flip)) / 2.0
        return F.softmax(logits, dim=1)

def ema_update(prev, cur, alpha):
    if prev is None: return cur
    return (1 - alpha) * prev + alpha * cur

# -------- detectors --------
class MPFaceDetector:
    def __init__(self, min_conf=0.6):
        try:
            import mediapipe as mp
        except ImportError:
            raise RuntimeError("mediapipe not installed. pip install mediapipe or use --backend opencv.")
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_conf)
    def __call__(self, frame_bgr):
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        boxes, scores = [], []
        if res.detections:
            for d in res.detections:
                s = d.score[0]
                bbox = d.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                ww, hh = int(bbox.width * w), int(bbox.height * h)
                boxes.append([x, y, x+ww, y+hh]); scores.append(s)
        return boxes, scores

class CV2DNNFaceDetector:
    def __init__(self, min_conf=0.6):
        self.min_conf = min_conf
        proto = Path("models/face_detector/deploy.prototxt")
        caff = Path("models/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        if not (proto.exists() and caff.exists()):
            raise RuntimeError("Missing OpenCV DNN model files. Use --backend mediapipe or place SSD files.")
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(caff))
    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 1.0, (300,300), (104,177,123))
        self.net.setInput(blob)
        dets = self.net.forward()
        boxes, scores = [], []
        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf < self.min_conf: continue
            box = dets[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            boxes.append([x1,y1,x2,y2]); scores.append(conf)
        return boxes, scores

# -------- main --------
def main():
    args = parse_args()
    labels = load_labels(); k = len(labels)

    # checkpoint
    ckpt = Path(args.ckpt) if args.ckpt else latest_best()
    assert ckpt is not None, "No *_best.pt found. Train first or pass --ckpt"
    run_tag = ckpt.stem.replace("_best","")
    backbone = infer_backbone(run_tag)
    print(f"[app] device={DEVICE} | run={run_tag} | backbone={backbone}")

    # model
    model = build_model(backbone, k).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # detector
    det = MPFaceDetector(args.min_conf) if args.backend == "mediapipe" else CV2DNNFaceDetector(args.min_conf)

    # capture
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    # writer
    writer = None
    if args.write:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    # smoothing state
    prob_ema = None
    total_frames, dur_det, dur_inf = 0, 0.0, 0.0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break
        total_frames += 1
        h, w = frame.shape[:2]

        # detect
        t_det = time.time()
        boxes, scores = det(frame)
        dur_det += time.time() - t_det

        # choose largest face (simple)
        face_label = "no-face"; face_prob = 0.0
        if boxes:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            idx = int(np.argmax(areas))
            x1,y1,x2,y2 = boxes[idx]
            x1,y1,x2,y2 = add_margin(x1,y1,x2,y2,w,h,args.margin)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (50,220,50), 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                x = preprocess_crop(crop, args.img_size).to(DEVICE)
                t_inf = time.time()
                probs = softmax_infer(model, x, use_tta=args.use_tta)[0].detach().cpu().numpy()
                dur_inf += time.time() - t_inf

                # EMA smoothing
                prob_ema = ema_update(prob_ema, probs, args.ema_alpha)
                probs_show = prob_ema if prob_ema is not None else probs

                pred_idx = int(np.argmax(probs_show))
                face_prob = float(probs_show[pred_idx])
                face_label = labels[pred_idx]
                if face_prob < args.unknown_thresh:
                    face_label = "unknown"

                # overlay
                txt = f"{face_label} {face_prob:.2f}"
                cv2.putText(frame, txt, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)

        # HUD
        elapsed = time.time() - t0
        fps = total_frames / max(1e-6, elapsed)
        cv2.putText(frame, f"FPS:{fps:4.1f} det:{args.backend} run:{run_tag}", (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,180), 2, cv2.LINE_AA)

        if args.display:
            cv2.imshow("DeepFER Realtime", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if writer is not None:
            writer.write(frame)

        if args.max_frames and total_frames >= args.max_frames:
            break

    cap.release()
    if writer is not None: writer.release()
    if args.display: cv2.destroyAllWindows()

    dur = time.time() - t0
    fps_avg = total_frames / max(1e-6, dur)
    print(f"[done] frames={total_frames} dur={dur:.1f}s fps_avg={fps_avg:.1f} "
          f"det_ms={(dur_det/max(1,total_frames))*1000:.1f} inf_ms={(dur_inf/max(1,total_frames))*1000:.1f}")

    # log
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run": run_tag, "backend": args.backend, "source": args.source,
        "frames": total_frames, "duration_s": round(dur,3), "fps_avg": round(fps_avg,2),
        "det_ms": round((dur_det/max(1,total_frames))*1000,2),
        "inf_ms": round((dur_inf/max(1,total_frames))*1000,2),
        "ema_alpha": args.ema_alpha, "unknown_thresh": args.unknown_thresh, "tta": args.use_tta
    }
    log_csv = METRICS_DIR / "latency_benchmark.csv"
    write_header = not log_csv.exists()
    with open(log_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        print("[FATAL]", e)
        traceback.print_exc()
        sys.exit(1)
