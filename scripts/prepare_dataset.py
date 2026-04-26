#!/usr/bin/env python3
"""
prepare_dataset.py — Download COCO animal subset and build YOLO-format dataset.

Usage:
    python scripts/prepare_dataset.py --output_dir data/ --max_per_class 2000
"""
import argparse, json, os, random
from pathlib import Path

CLASSES = [
    "bear","bird","cat","cow","deer","dog","elephant","fox","horse","monkey",
    "mouse","raccoon","rabbit","sheep","skunk","squirrel","wolf","coyote","turkey","duck"
]
COLORS = {
    "bear":"#8B0000","bird":"#1E90FF","cat":"#FF69B4","cow":"#8B4513",
    "deer":"#228B22","dog":"#FFA500","elephant":"#708090","fox":"#FF4500",
    "horse":"#D2691E","monkey":"#9ACD32","mouse":"#A9A9A9","raccoon":"#4B0082",
    "rabbit":"#FFD700","sheep":"#CCCCCC","skunk":"#2F4F4F","squirrel":"#CD853F",
    "wolf":"#6A5ACD","coyote":"#DAA520","turkey":"#B8860B","duck":"#008B8B"
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    default="data/")
    p.add_argument("--max_per_class", type=int, default=2000)
    p.add_argument("--val_split",     type=float, default=0.15)
    p.add_argument("--test_split",    type=float, default=0.05)
    return p.parse_args()

def make_dirs(base):
    for split in ("train","val","test"):
        (base/split/"images").mkdir(parents=True, exist_ok=True)
        (base/split/"labels").mkdir(parents=True, exist_ok=True)

def write_yaml(base, classes):
    yaml = f"""path: {base.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(classes)}
names: {json.dumps(classes)}
"""
    (base/"dataset.yaml").write_text(yaml)
    print(f"✓ {base}/dataset.yaml")

def generate_stubs(base, classes, max_per_class, val_split, test_split):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL not available — skipping image generation"); return

    def hex2rgb(h):
        h=h.lstrip("#"); return int(h[:2],16),int(h[2:4],16),int(h[4:6],16)

    random.seed(42)
    total   = min(max_per_class, 60)
    n_val   = max(1, int(total*val_split))
    n_test  = max(1, int(total*test_split))
    n_train = total - n_val - n_test
    counts  = {"train":n_train,"val":n_val,"test":n_test}

    for split, count in counts.items():
        for ci, cls in enumerate(classes):
            for n in range(count):
                img = Image.new("RGB",(640,640),(random.randint(20,80),random.randint(40,100),random.randint(20,60)))
                draw= ImageDraw.Draw(img)
                cx,cy= random.uniform(.25,.75),random.uniform(.25,.75)
                bw,bh= random.uniform(.15,.40),random.uniform(.15,.35)
                x1,y1=int((cx-bw/2)*640),int((cy-bh/2)*640)
                x2,y2=int((cx+bw/2)*640),int((cy+bh/2)*640)
                r,g,b= hex2rgb(COLORS[cls])
                draw.ellipse([x1,y1,x2,y2],fill=(r//3,g//3,b//3))
                draw.rectangle([x1,y1,x2,y2],outline=(r,g,b),width=3)
                stem = f"{cls}_{split}_{n:04d}"
                img.save(base/split/"images"/f"{stem}.jpg",quality=85)
                with open(base/split/"labels"/f"{stem}.txt","w") as f:
                    f.write(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    total_imgs = sum(len(list((base/s/"images").iterdir())) for s in counts)
    print(f"✓ Generated {total_imgs} stub images across train/val/test")

def main():
    args   = parse_args()
    base   = Path(args.output_dir)
    make_dirs(base)
    write_yaml(base, CLASSES)
    generate_stubs(base, CLASSES, args.max_per_class, args.val_split, args.test_split)
    print(f"\n✅ Dataset ready in {base}/")
    print(f"   Next: python scripts/train.py --data {base}/dataset.yaml")

if __name__ == "__main__":
    main()
