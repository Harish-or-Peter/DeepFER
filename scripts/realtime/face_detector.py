import argparse, time, csv
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
(PRED_DIR / "crops").mkdir(parents=True, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0",
                    help="0 for webcam, or path to video file")
    ap.add_argument("--backend", type=str, default="mediapipe",
                    choices=["mediapipe", "opencv"],
                    help="Face detection backend")
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--margin", type=float, default=0.15,
                    help="Margin around bbox when saving crops (fraction of max side)")
    ap.add_argument("--display", action="store_true",
                    help="Show a live window")
    ap.add_argument("--write", action="store_true",
                    help="Write annotated MP4 to results/predictions/demo.mp4")
    ap.add_argument("--save_crops", action="store_true",
                    help="Save face crops to results/predictions/crops/<run>/")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Stop after N frames (0 = no limit)")
    ap.add_argument("--out", type=str, default=str(PRED_DIR / "demo.mp4"))
    return ap.parse_args()

# -------- MediaPipe wrapper --------
class MPFaceDetector:
    def __init__(self, min_conf=0.6):
        try:
            import mediapipe as mp
        except ImportError:
            raise RuntimeError("mediapipe is not installed. pip install mediapipe or use --backend opencv.")
        self.mp = mp
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_conf)

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        boxes, scores = [], []
        if res.detections:
            for d in res.detections:
                score = d.score[0]
                bbox = d.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                ww, hh = int(bbox.width * w), int(bbox.height * h)
                boxes.append([x, y, x + ww, y + hh])
                scores.append(score)
        return boxes, scores

# -------- OpenCV DNN wrapper (SSD) --------
class CV2DNNFaceDetector:
    def __init__(self, min_conf=0.6):
        self.min_conf = min_conf
        # Expect model assets in models/face_detector/
        proto = Path("models/face_detector/deploy.prototxt")
        caff = Path("models/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        if not (proto.exists() and caff.exists()):
            raise RuntimeError(
                f"OpenCV DNN requires model files.\n"
                f"Place them at:\n{proto}\n{caff}\n"
                f"Or use --backend mediapipe instead."
            )
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(caff))

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)),
                                     1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        dets = self.net.forward()
        boxes, scores = [], []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self.min_conf: continue
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
        return boxes, scores

def clamp(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)

def add_margin(x1, y1, x2, y2, w, h, mfrac):
    bw, bh = x2 - x1, y2 - y1
    m = int(max(bw, bh) * mfrac)
    return clamp(x1 - m, y1 - m, x2 + m, y2 + m, w, h)

def main():
    args = parse_args()
    run_tag = datetime.now().strftime("fd_%Y%m%d_%H%M%S")

    # Capture
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    # Detector
    if args.backend == "mediapipe":
        det = MPFaceDetector(min_conf=args.min_conf)
    else:
        det = CV2DNNFaceDetector(min_conf=args.min_conf)

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    if args.write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Crops dir
    crop_dir = (PRED_DIR / "crops" / run_tag) if args.save_crops else None
    if crop_dir: crop_dir.mkdir(parents=True, exist_ok=True)

    total_frames, frames_with_face = 0, 0
    t0 = time.time()
    fps_smoothed, alpha = 0.0, 0.1

    while True:
        ok, frame = cap.read()
        if not ok: break
        total_frames += 1
        h, w = frame.shape[:2]

        t_det = time.time()
        boxes, scores = det(frame)
        dt = time.time() - t_det

        has_face = len(boxes) > 0
        if has_face: frames_with_face += 1

        # Draw & save crops
        for i, (b, s) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = b
            x1, y1, x2, y2 = clamp(x1, y1, x2, y2, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 50), 2)
            label = f"face {s:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 200, 20), 1, cv2.LINE_AA)

            if crop_dir:
                mx1, my1, mx2, my2 = add_margin(x1, y1, x2, y2, w, h, args.margin)
                crop = frame[my1:my2, mx1:mx2]
                if crop.size > 0:
                    cv2.imwrite(str(crop_dir / f"f{total_frames:06d}_{i}.jpg"), crop)

        # FPS annotate
        fps_cur = 1.0 / max(1e-6, dt)
        fps_smoothed = fps_cur if fps_smoothed == 0 else (1 - alpha) * fps_smoothed + alpha * fps_cur
        cv2.putText(frame, f"FPS:{fps_smoothed:5.1f}  dets:{len(boxes)}  backend:{args.backend}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        if args.display:
            cv2.imshow("FaceDetector", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        if writer is not None:
            writer.write(frame)

        if args.max_frames and total_frames >= args.max_frames:
            break

    cap.release()
    if writer is not None: writer.release()
    if args.display: cv2.destroyAllWindows()

    dur = time.time() - t0
    det_rate = frames_with_face / max(1, total_frames)
    fps_avg = total_frames / max(1e-6, dur)

    print(f"[done] frames={total_frames} duration={dur:.1f}s fps_avg={fps_avg:.1f} det_rate={det_rate:.3f}")
    if args.write:
        print(f"[saved] video → {args.out}")
    if crop_dir:
        print(f"[saved] crops → {crop_dir}")

    # Log to CSV (append)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "source": args.source,
        "frames": total_frames,
        "duration_s": round(dur, 3),
        "fps_avg": round(fps_avg, 2),
        "det_rate": round(det_rate, 3)
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