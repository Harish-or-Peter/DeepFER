
# DeepFER — Facial Emotion Recognition

Real-time facial emotion recognition (7 classes) with PyTorch/ONNX, MediaPipe face detection, Grad-CAM explainability, robustness tests, and a webcam web app.

**Best model:** EfficientNet-B0
**Test accuracy:** \~**0.7008**
**ONNX parity:** identical accuracy to PyTorch on a 500-image test slice (mean |Δprob| ≈ **1.3e-4**)
**Live demo:** FastAPI backend + `web/index.html` client

---

## 1) Quickstart

### 1.1 Install

```bash
# (recommended) create venv/conda first
pip install -r requirements.txt
```

GPU (optional):

```bash
python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

### 1.2 Dataset layout

```
data/
├─ train/<class>/*.jpg
├─ val/<class>/*.jpg
├─ test/<class>/*.jpg
└─ metadata/classes.json         # {"label_to_index": {"angry":0, "disgust":1, ...}}
```

> Run the audit once:

```bash
python -u scripts/data_audit.py
```

Creates `results/metrics/data_audit_per_class.csv` used for class weights.

---

## 2) Train (transfer learning)

### EfficientNet-B0 (PowerShell single line)

```powershell
python -u scripts/train_transfer.py --backbone efficientnet_b0 --epochs_head 5 --epochs_ft 25 --lr_ft 2e-4 --batch_size 48 --use_tta --use_mixup --mixup_alpha 0.1 --patience 8
```

### EfficientNet-B0 (bash)

```bash
python -u scripts/train_transfer.py \
  --backbone efficientnet_b0 \
  --epochs_head 5 \
  --epochs_ft 25 \
  --lr_ft 2e-4 \
  --batch_size 48 \
  --use_tta --use_mixup --mixup_alpha 0.1 \
  --patience 8
```

ResNet-50 is also supported (kept for your presentation): `--backbone resnet50`.

**Outputs**

* Checkpoints → `models/checkpoints/<run>_*` (`*_best.pt`, `*_after_head.pt`, per-epoch if enabled)
* Reports → `results/metrics/<run>_*`
* TensorBoard → `results/runs/<run>/`

Resume (optional):

```bash
python -u scripts/train_transfer.py --backbone efficientnet_b0 --resume models/checkpoints/<your_run>_last.pt
```

---

## 3) Evaluate & Explain

### 3.1 Error analysis (reports + confusions)

```bash
python -u scripts/error_analysis.py
```

### 3.2 Robustness (brightness/blur/rotation/noise)

```bash
python -u scripts/robustness_eval.py
```

### 3.3 Grad-CAM (side-by-side grids)

```bash
python -u scripts/gradcam.py --run <your_run_name>
```

Images land in `results/predictions/gradcam/`.

---

## 4) Real-time demo (FastAPI + Web)

### 4.1 Start backend (FastAPI)

```bash
uvicorn scripts.serve.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4.2 Open the web page

Serve locally so the browser allows camera:

```bash
python -m http.server 5500
```

Open: `http://127.0.0.1:5500/web/index.html`

**Tips**

* Set **Backend URL** to `http://127.0.0.1:8000`
* Toggle **Face detect** to crop with MediaPipe
* Adjust **Unknown threshold** (e.g., 0.35–0.55)
* Use **ONNX** engine (fastest)
* Snapshot writes to `results/predictions/`

---

## 5) ONNX & Latency

Export is automatic in the backend; you can also run parity and latency checks:

**Accuracy parity (PyTorch vs ONNX)**

```bash
python -u scripts/bench/accuracy_compare_onnx.py --split test --max_images 500
```

**Latency benchmark**
Results saved to `results/metrics/latency_benchmark.csv`. We observed **≥30%** reduction vs eager PyTorch with ONNXRuntime.

---

## 6) Final artifacts & reproducibility

After your best run, copy to a clean bundle:

```
models/final_model/
├─ model_best.pt
├─ best_config.yaml              # your frozen config (included in this repo)
└─ manifest.txt                  # run id, timestamp, labels, transforms
```

You can also keep `model.onnx` here for deployment.

---

## 7) Repo map

```
deepfer_project/
├─ data/
│  └─ metadata/classes.json
├─ models/
│  ├─ checkpoints/
│  └─ final_model/
├─ results/
│  ├─ metrics/
│  └─ predictions/gradcam/
├─ scripts/
│  ├─ data_preprocessing.py
│  ├─ data_audit.py
│  ├─ train_transfer.py
│  ├─ error_analysis.py
│  ├─ robustness_eval.py
│  ├─ gradcam.py
│  ├─ bench/accuracy_compare_onnx.py
│  └─ serve/api.py
└─ web/
   └─ index.html
```

---

## 8) Known results (for your report/slides)

* **EfficientNet-B0:** Val ≈ **0.6959**, Test ≈ **0.7008**
* **Robustness:** ok under mild brightness/blur; sensitive to rotation ≥20–30° and strong noise
* **ONNX Parity:** PyTorch acc 0.718, ONNX acc 0.718 on 500-image test slice

---

## 9) Troubleshooting

* **404 for `/index.html`** → Start the static server in repo root: `python -m http.server 5500` and open `/web/index.html`.
* **Camera blocked** → Allow browser camera permission; use `https` or `http://localhost`.
* **Black right canvas** → Click **Start**; ensure Backend URL is correct; check FastAPI logs.
* **Predicting when no face** → Enable **Face detect** and/or raise **Unknown threshold**.
* **Dominant “surprise”** → Use class-prior calibration (enabled server-side) and set Unknown threshold around 0.4–0.5.
* **CUDA OOM** → Reduce batch size (e.g., `--batch_size 32`) or fall back to CPU.

---

## 10) Citation

If this project helps your research/work, please cite the standard papers:

* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
* Tan & Le, *EfficientNet*, ICML 2019
* Zhang et al., *mixup*, ICLR 2018
* Szegedy et al., *Rethinking the Inception Architecture* (label smoothing), CVPR 2016
* MediaPipe Face Detection

---

## 11) License

Add your preferred license (MIT/BSD/Apache-2.0) to `LICENSE`.

---

### One-liners you’ll reuse

**Train (best E-B0):**

```powershell
python -u scripts/train_transfer.py --backbone efficientnet_b0 --epochs_head 5 --epochs_ft 25 --lr_ft 2e-4 --batch_size 48 --use_tta --use_mixup --mixup_alpha 0.1 --patience 8
```

**Serve backend:**

```powershell
uvicorn scripts.serve.api:app --host 0.0.0.0 --port 8000 --reload
```

**Open web UI:**

```powershell
python -m http.server 5500
```

**Parity check:**

```powershell
python -u scripts/bench/accuracy_compare_onnx.py --split test --max_images 500
```

