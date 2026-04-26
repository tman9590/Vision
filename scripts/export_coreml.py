#!/usr/bin/env python3
"""
export_coreml.py — Convert trained best.pt → AnimalDetector.mlpackage

Requires macOS 13+ with coremltools installed.

Usage:
    python scripts/export_coreml.py --weights runs/train/weights/best.pt
    python scripts/export_coreml.py --weights runs/train/weights/best.pt --half
"""
import argparse, json, os, shutil, sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
CONFIG     = ROOT / "config.json"
MODEL_DIR  = ROOT / "model"

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics coremltools")

    with open(CONFIG) as f:
        cfg = json.load(f)

    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--output",  default=str(MODEL_DIR))
    p.add_argument("--imgsz",   type=int, default=cfg["model"]["input_size"][0])
    p.add_argument("--half",    action="store_true")
    p.add_argument("--conf",    type=float, default=cfg["model"]["confidence_threshold"])
    p.add_argument("--iou",     type=float, default=cfg["model"]["iou_threshold"])
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    print(f"Loading {weights} ...")
    model = YOLO(str(weights))

    print("Exporting to CoreML ...")
    out = model.export(
        format="coreml", imgsz=args.imgsz,
        half=args.half, nms=cfg["export"]["nms"],
        conf=args.conf, iou=args.iou,
        simplify=cfg["export"]["simplify"],
    )

    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    final = dst / cfg["model"]["coreml_file"]
    if Path(out) != final:
        if final.exists():
            shutil.rmtree(final)
        shutil.copytree(out, final)

    # Annotate metadata
    try:
        import coremltools as ct
        mlm = ct.models.MLModel(str(final))
        mlm.user_defined_metadata["classes"]     = json.dumps(cfg["classes"])
        mlm.user_defined_metadata["conf_thresh"] = str(args.conf)
        mlm.user_defined_metadata["iou_thresh"]  = str(args.iou)
        mlm.user_defined_metadata["input_size"]  = json.dumps(cfg["model"]["input_size"])
        mlm.save(str(final))
    except Exception as e:
        print(f"  Metadata annotation skipped: {e}")

    total = sum(os.path.getsize(os.path.join(r,f))
                for r,_,fs in os.walk(final) for f in fs)
    print(f"\n✅ CoreML model saved: {final}  ({total/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
