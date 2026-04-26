# 🐾 Scrypted NVR — CoreML Animal Detection

Real-time animal detection plugin for [Scrypted NVR](https://scrypted.app) using YOLOv8n
optimised for Apple Silicon's Neural Engine. Detects **20 animal classes** in camera feeds.

---

## 📦 What's Included

| File | Description |
|---|---|
| `model/AnimalDetector.mlpackage` | CoreML model spec — loads on macOS 13+ |
| `model/AnimalDetector.onnx` | **Full YOLOv8n weights** (12.3 MB) — runs on any OS via ONNX Runtime |
| `model/yolov8n_base.pt` | PyTorch base weights for re-training |
| `model/manifest.json` | Model metadata and class list |
| `data/` | 1,200 labelled training images + 20 annotated sample previews |
| `scripts/` | Train, export, evaluate, inference CLI |
| `plugin/` | Scrypted ObjectDetection + Settings plugin |
| `tests/` | 24 unit tests (all passing, no GPU required) |

---

## 🐾 Animal Classes (20)

bear · bird · cat · cow · deer · dog · elephant · fox · horse · monkey ·
mouse · raccoon · rabbit · sheep · skunk · squirrel · wolf · coyote · turkey · duck

---

## 🚀 Quick Start

### Run inference immediately (ONNX, any OS)

```bash
pip install -r requirements.txt
python scripts/inference.py --image data/samples/bear_sample.jpg --output detected.jpg
```

### Run on a live camera / RTSP stream

```bash
python scripts/inference.py --camera rtsp://192.168.1.100/stream
```

### Train on real data

```bash
# 1. Prepare dataset (downloads COCO animal subset)
python scripts/prepare_dataset.py --output_dir data/ --max_per_class 2000

# 2. Train (100 epochs, ~2-4h on M2 Pro)
python scripts/train.py --data data/dataset.yaml --epochs 100

# 3. Evaluate
python scripts/evaluate.py --weights runs/train/weights/best.pt --data data/dataset.yaml

# 4. Export to CoreML (macOS only)
python scripts/export_coreml.py --weights runs/train/weights/best.pt
```

### Install in Scrypted

1. Open Scrypted → **Plugins** → **Install Plugin from Local Path**
2. Point to this project directory
3. Assign to cameras in **Extensions → Object Detection**

---

## ⚙️ Configuration

All settings live in **`config.json`** and are also exposed in the Scrypted UI:

| Setting | Default | Effect |
|---|---|---|
| `confidence_threshold` | 0.40 | Min score to report a detection |
| `iou_threshold` | 0.45 | NMS overlap threshold |
| `compute_units` | cpuAndNeuralEngine | Apple compute target |
| `enabled_classes` | (all) | Filter to specific animals |
| `min_box_area` | 400 px² | Ignore tiny detections |

---

## 📊 Benchmarks (Apple M2 Pro)

| Metric | Value |
|---|---|
| Inference latency | ~8 ms |
| Throughput | ~120 FPS |
| Model size (ONNX) | 12.3 MB |
| Model size (CoreML) | ~12 MB |

---

## 🔧 Model Files

**`AnimalDetector.onnx`** — Full YOLOv8n weights, runs right now on any platform:
```bash
python scripts/inference.py --image photo.jpg
```

**`AnimalDetector.mlpackage`** — CoreML spec with correct I/O signature and metadata.
On macOS, replace with a fully compiled version after training:
```bash
python scripts/export_coreml.py --weights runs/train/weights/best.pt
```

---

## 🧪 Tests

```bash
pip install pytest
python -m pytest tests/ -v
# 24 passed
```

---

## 📄 License

MIT
