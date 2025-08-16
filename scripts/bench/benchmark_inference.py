import argparse, json, time, statistics
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DATA_DIR = Path("data")
CKPT_DIR = Path("models/checkpoints")
EXP_DIR  = Path("models/export")
METRICS_DIR = Path("results/metrics"); METRICS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

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

def timeit_ms(fn, warmup=30, runs=200):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "p90_ms": np.percentile(times, 90),
        "min_ms": min(times),
        "max_ms": max(times),
        "runs": runs
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=str, required=True, choices=["pytorch","torchscript","onnxruntime"])
    ap.add_argument("--ckpt", type=str, default=None, help="for pytorch engine")
    ap.add_argument("--ts", type=str, default=str(EXP_DIR/"model_ts.pt"))
    ap.add_argument("--onnx", type=str, default=str(EXP_DIR/"model.onnx"))
    ap.add_argument("--provider", type=str, default="auto", help="onnxruntime provider: auto|cuda|cpu")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    k = len(load_labels())
    x = torch.randn(args.batch_size,3,args.img_size,args.img_size)

    if args.engine == "pytorch":
        ckpt = Path(args.ckpt) if args.ckpt else latest_best()
        assert ckpt is not None, "No *_best.pt found"
        run_tag = ckpt.stem.replace("_best","")
        backbone = infer_backbone(run_tag)
        model = build_model(backbone, k).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        x = x.to(DEVICE)

        def fn():
            with torch.no_grad():
                _ = model(x)

        stats = timeit_ms(fn, warmup=args.warmup, runs=args.runs)
        engine_name = f"pytorch-{backbone}"

    elif args.engine == "torchscript":
        ts_path = Path(args.ts); assert ts_path.exists(), f"{ts_path} not found"
        model = torch.jit.load(str(ts_path), map_location=DEVICE).eval()
        x = x.to(DEVICE)

        def fn():
            with torch.no_grad():
                _ = model(x)

        stats = timeit_ms(fn, warmup=args.warmup, runs=args.runs)
        engine_name = "torchscript"

    else:
        import onnxruntime as ort
        onnx_path = Path(args.onnx); assert onnx_path.exists(), f"{onnx_path} not found"
        providers = ort.get_available_providers()
        if args.provider == "cuda" and "CUDAExecutionProvider" in providers:
            provider = ["CUDAExecutionProvider","CPUExecutionProvider"]
        elif args.provider == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            provider = providers
        sess = ort.InferenceSession(str(onnx_path), providers=provider)
        inp_name = sess.get_inputs()[0].name

        # ORT expects numpy
        x_np = x.numpy().astype(np.float32)

        def fn():
            _ = sess.run(None, {inp_name: x_np})

        stats = timeit_ms(fn, warmup=args.warmup, runs=args.runs)
        engine_name = f"onnxruntime({sess.get_providers()[0]})"

    fps = 1000.0 / stats["p50_ms"] * args.batch_size
    print(f"[bench] engine={engine_name} bs={args.batch_size} "
          f"p50={stats['p50_ms']:.2f}ms p90={stats['p90_ms']:.2f}ms mean={stats['mean_ms']:.2f}ms "
          f"fps≈{fps:.1f}")

    # log CSV
    import csv
    row = {
        "engine": engine_name,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "p50_ms": round(stats["p50_ms"],2),
        "p90_ms": round(stats["p90_ms"],2),
        "mean_ms": round(stats["mean_ms"],2),
        "fps_est": round(fps,1),
        "runs": stats["runs"]
    }
    csv_path = METRICS_DIR / "latency_benchmark.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

if __name__ == "__main__":
    main()
