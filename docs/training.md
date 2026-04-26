# Training Guide

## Setup

```bash
pip install ultralytics coremltools opencv-python-headless numpy pillow
```

## Prepare Dataset

```bash
python scripts/prepare_dataset.py --output_dir data/ --max_per_class 2000
```

Generates `data/dataset.yaml` + train/val/test splits in YOLO format.
To use real data, add annotated images to `data/train/images/` with matching `.txt` label files.

Each label row: `<class_id> <cx> <cy> <w> <h>` (all normalised 0–1).
Class IDs match the order in `config.json → classes`.

## Train

```bash
# Full run (~100 epochs, 2-4h on Apple M2 Pro)
python scripts/train.py --data data/dataset.yaml --epochs 100 --batch 16 --device mps

# Resume interrupted run
python scripts/train.py --resume runs/train/weights/last.pt
```

Checkpoints → `runs/train/weights/best.pt`

## Evaluate

```bash
python scripts/evaluate.py --weights runs/train/weights/best.pt --data data/dataset.yaml
```

Prints per-class mAP table and saves `runs/eval/results/results.json`.

## Export to CoreML (macOS only)

```bash
python scripts/export_coreml.py --weights runs/train/weights/best.pt
```

Saves `model/AnimalDetector.mlpackage` — drop this into the plugin and restart Scrypted.

## Expected Results (100 epochs, 2000 images/class)

| Metric | Expected |
|---|---|
| mAP@0.5 | 0.70 – 0.78 |
| mAP@0.5:0.95 | 0.48 – 0.56 |
| Inference (M2 Neural Engine) | ~8 ms |
