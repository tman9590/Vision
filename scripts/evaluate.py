#!/usr/bin/env python3
"""
evaluate.py — Per-class mAP, precision, recall on the test split.

Usage:
    python scripts/evaluate.py --weights runs/train/weights/best.pt --data data/dataset.yaml
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics")

    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data",    required=True)
    p.add_argument("--split",   default="test", choices=["val","test"])
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--output",  default="runs/eval")
    args = p.parse_args()

    with open(ROOT/"config.json") as f:
        cfg = json.load(f)

    model   = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split,
                        imgsz=args.imgsz, batch=args.batch,
                        plots=True, save_json=True,
                        project=args.output, name="results")

    classes = cfg["classes"]
    box     = metrics.box

    print("\n" + "═"*65)
    print(f"{'CLASS':<14}  {'P':>8}  {'R':>8}  {'mAP50':>8}  {'F1':>6}")
    print("─"*65)
    for i, cls in enumerate(classes):
        try:
            pp = float(box.p[i]); rr = float(box.r[i]); ap = float(box.ap[i])
            f1 = 2*pp*rr/(pp+rr+1e-9)
            print(f"{cls:<14}  {pp:>8.4f}  {rr:>8.4f}  {ap:>8.4f}  {f1:>6.4f}")
        except Exception:
            print(f"{cls:<14}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>6}")
    print("─"*65)
    print(f"{'ALL':<14}  {float(box.mp):>8.4f}  {float(box.mr):>8.4f}  {float(box.map50):>8.4f}")
    print(f"\nmAP@0.5:0.95 = {float(box.map):.4f}")
    print("═"*65)

    out = Path(args.output)/"results"/"results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "overall": {"mAP50": float(box.map50), "mAP50_95": float(box.map),
                    "precision": float(box.mp), "recall": float(box.mr)},
    }, indent=2))
    print(f"\n✓ Results saved to {out}")

if __name__ == "__main__":
    main()
