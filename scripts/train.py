#!/usr/bin/env python3
"""
train.py — Fine-tune YOLOv8n on the 20-class animal dataset.

Usage:
    python scripts/train.py --data data/dataset.yaml --epochs 100
    python scripts/train.py --resume runs/train/weights/last.pt
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics")

    with open(ROOT / "config.json") as f:
        t = json.load(f)["training"]

    p = argparse.ArgumentParser()
    p.add_argument("--data",     default="data/dataset.yaml")
    p.add_argument("--model",    default=t["base_model"])
    p.add_argument("--epochs",   type=int, default=t["epochs"])
    p.add_argument("--imgsz",    type=int, default=t["imgsz"])
    p.add_argument("--batch",    type=int, default=t["batch"])
    p.add_argument("--patience", type=int, default=t["patience"])
    p.add_argument("--device",   default="")
    p.add_argument("--project",  default="runs")
    p.add_argument("--name",     default="train")
    p.add_argument("--resume",   default=None)
    p.add_argument("--exist_ok", action="store_true")
    args = p.parse_args()

    model = YOLO(args.resume or args.model)

    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, patience=args.patience,
        optimizer=t["optimizer"], lr0=t["lr0"], lrf=t["lrf"],
        momentum=t["momentum"], weight_decay=t["weight_decay"],
        augment=t["augment"], mosaic=t["mosaic"], mixup=t["mixup"],
        copy_paste=t["copy_paste"], hsv_h=t["hsv_h"], hsv_s=t["hsv_s"],
        hsv_v=t["hsv_v"], degrees=t["degrees"], translate=t["translate"],
        scale=t["scale"], fliplr=t["fliplr"], flipud=t["flipud"],
        project=args.project, name=args.name,
        device=args.device or None, exist_ok=args.exist_ok,
        plots=True, save=True, val=True, verbose=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n✅ Training complete — best checkpoint: {best}")
    print(f"Next: python scripts/export_coreml.py --weights {best}")

if __name__ == "__main__":
    main()
