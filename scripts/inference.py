#!/usr/bin/env python3
"""
inference.py — CoreML / ONNX animal detection engine for Scrypted NVR.

Usage:
    python scripts/inference.py --image /path/to/frame.jpg --output out.jpg
    python scripts/inference.py --camera rtsp://192.168.1.100/stream
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import cv2, numpy as np

ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.json"
MODEL_DIR   = ROOT / "model"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_name: str
    class_id:   int
    confidence: float
    bbox:       tuple[float, float, float, float]   # x1 y1 x2 y2 pixels
    bbox_norm:  tuple[float, float, float, float]   # 0-1

    def to_dict(self):
        return {"class": self.class_name, "class_id": self.class_id,
                "confidence": round(float(self.confidence), 4),
                "bbox": [round(v,1) for v in self.bbox],
                "bbox_norm": [round(v,6) for v in self.bbox_norm]}

    def to_scrypted(self):
        x1,y1,x2,y2 = self.bbox
        return {"className": self.class_name, "score": float(self.confidence),
                "boundingBox": {"x":x1,"y":y1,"width":x2-x1,"height":y2-y1}}


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    image_w: int = 0
    image_h: int = 0

    @property
    def animals_present(self): return sorted({d.class_name for d in self.detections})

    def to_dict(self):
        return {"detections": [d.to_dict() for d in self.detections],
                "animals_present": self.animals_present,
                "count": len(self.detections),
                "latency_ms": round(self.latency_ms, 2)}


# ── Preprocessor ─────────────────────────────────────────────────────────────

class Preprocessor:
    def __init__(self, target: int = 640):
        self.target = target

    def __call__(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        scale  = self.target / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pad_w   = self.target - new_w
        pad_h   = self.target - new_h
        padded  = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_CONSTANT, value=(114,114,114))
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2,0,1))[np.newaxis]
        return blob, scale, (pad_w, pad_h)


# ── NMS ───────────────────────────────────────────────────────────────────────

def xywh2xyxy(b):
    o = np.empty_like(b)
    o[...,0] = b[...,0] - b[...,2]/2
    o[...,1] = b[...,1] - b[...,3]/2
    o[...,2] = b[...,0] + b[...,2]/2
    o[...,3] = b[...,1] + b[...,3]/2
    return o

def nms(boxes, scores, iou_thresh):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]; keep.append(int(i))
        if order.size == 1: break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou   = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


# ── Postprocessor ─────────────────────────────────────────────────────────────

class Postprocessor:
    def __init__(self, classes, conf_thresh, iou_thresh):
        self.classes     = classes
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

    def __call__(self, raw, orig_w, orig_h, scale, padding):
        pred = raw[0] if raw.ndim == 3 else raw
        if pred.shape[0] < pred.shape[1]: pred = pred.T
        nc       = len(self.classes)
        box_xywh = pred[:, :4]
        cls_conf = pred[:, 4:4+nc]
        scores   = cls_conf.max(axis=1)
        cls_ids  = cls_conf.argmax(axis=1)
        mask     = scores >= self.conf_thresh
        box_xywh, scores, cls_ids = box_xywh[mask], scores[mask], cls_ids[mask]
        if not len(scores): return []
        boxes_lb = xywh2xyxy(box_xywh)
        keep_all = []
        for cid in np.unique(cls_ids):
            idx  = np.where(cls_ids == cid)[0]
            kept = nms(boxes_lb[idx], scores[idx], self.iou_thresh)
            keep_all.extend(idx[kept])
        dets = []
        pad_w, pad_h = padding
        for i in keep_all:
            x1 = boxes_lb[i,0] / scale
            y1 = boxes_lb[i,1] / scale
            x2 = (boxes_lb[i,2] - pad_w) / scale
            y2 = (boxes_lb[i,3] - pad_h) / scale
            x1,y1 = max(0.,x1), max(0.,y1)
            x2,y2 = min(x2,orig_w), min(y2,orig_h)
            if (x2-x1)*(y2-y1) < 1: continue
            cid  = int(cls_ids[i])
            name = self.classes[cid] if cid < len(self.classes) else f"cls_{cid}"
            dets.append(Detection(
                class_name = name, class_id = cid,
                confidence = float(scores[i]),
                bbox       = (x1,y1,x2,y2),
                bbox_norm  = (x1/orig_w, y1/orig_h, x2/orig_w, y2/orig_h),
            ))
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets


# ── Detector ──────────────────────────────────────────────────────────────────

class AnimalDetector:
    """
    Loads AnimalDetector.mlpackage (CoreML, macOS) or AnimalDetector.onnx
    (ONNX Runtime, any OS) automatically.
    """

    def __init__(self, model_path=None, config_path=None,
                 conf_threshold=None, iou_threshold=None, compute_units=None):
        with open(config_path or CONFIG_PATH) as f:
            self.cfg = json.load(f)
        self.classes     = self.cfg["classes"]
        self.conf_thresh = conf_threshold or self.cfg["model"]["confidence_threshold"]
        self.iou_thresh  = iou_threshold  or self.cfg["model"]["iou_threshold"]
        self.compute     = compute_units  or self.cfg["model"]["compute_units"]
        self.input_size  = self.cfg["model"]["input_size"][0]
        self._model      = None
        self._backend    = None
        self._load(model_path)
        self.pre  = Preprocessor(self.input_size)
        self.post = Postprocessor(self.classes, self.conf_thresh, self.iou_thresh)

    def _load(self, model_path):
        mlpkg  = Path(model_path or MODEL_DIR / self.cfg["model"]["coreml_file"])
        onnx_p = Path(model_path or MODEL_DIR / self.cfg["model"]["onnx_file"])
        if not onnx_p.exists() and not mlpkg.exists():
            raise FileNotFoundError(f"No model found in {MODEL_DIR}")

        # Try CoreML first (macOS only)
        try:
            import coremltools as ct
            cu_map = {"cpuOnly": ct.ComputeUnit.CPU_ONLY,
                      "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
                      "all": ct.ComputeUnit.ALL,
                      "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE}
            cu = cu_map.get(self.compute, ct.ComputeUnit.CPU_AND_NE)
            self._model   = ct.models.MLModel(str(mlpkg), compute_units=cu)
            self._backend = "coreml"
            print(f"✓ CoreML [{self.compute}]  {mlpkg.name}")
            return
        except Exception as e:
            print(f"  CoreML unavailable ({type(e).__name__}), using ONNX Runtime")

        # ONNX Runtime fallback
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model   = ort.InferenceSession(str(onnx_p), providers=providers)
        self._backend = "onnx"
        print(f"✓ ONNX Runtime  {onnx_p.name}")

    def _infer(self, blob):
        if self._backend == "coreml":
            out = self._model.predict({self.cfg["model"]["input_name"]: blob})
            return np.array(out[self.cfg["model"]["output_name"]])
        else:
            name = self._model.get_inputs()[0].name
            return self._model.run(None, {name: blob})[0]

    def detect(self, image: np.ndarray) -> InferenceResult:
        if image is None or image.size == 0:
            raise ValueError("Empty image")
        h, w = image.shape[:2]
        t0   = time.perf_counter()
        blob, scale, padding = self.pre(image)
        raw  = self._infer(blob)
        dets = self.post(raw, w, h, scale, padding)
        ms   = (time.perf_counter() - t0) * 1000
        return InferenceResult(detections=dets, latency_ms=ms, image_w=w, image_h=h)

    def detect_path(self, path) -> InferenceResult:
        img = cv2.imread(str(path))
        if img is None: raise IOError(f"Cannot read: {path}")
        return self.detect(img)

    def draw(self, image: np.ndarray, result: InferenceResult) -> np.ndarray:
        colors = self.cfg.get("class_colors", {})
        out    = image.copy()
        for det in result.detections:
            x1,y1,x2,y2 = [int(v) for v in det.bbox]
            h  = colors.get(det.class_name,"#00FF00").lstrip("#")
            c  = (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))
            cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
            lbl = f"{det.class_name} {det.confidence:.0%}"
            (tw,th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1,y1-th-bl-4), (x1+tw,y1), c, -1)
            cv2.putText(out, lbl, (x1,y1-bl-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{result.latency_ms:.1f} ms", (10,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",   help="Image path")
    p.add_argument("--camera",  help="Camera index or RTSP URL")
    p.add_argument("--output",  help="Save annotated image here")
    p.add_argument("--conf",    type=float)
    p.add_argument("--iou",     type=float)
    p.add_argument("--json",    action="store_true")
    args = p.parse_args()

    det = AnimalDetector(conf_threshold=args.conf, iou_threshold=args.iou)

    if args.image:
        r = det.detect_path(args.image)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2))
        else:
            print(f"{len(r.detections)} detection(s)  {r.latency_ms:.1f} ms")
            for d in r.detections:
                print(f"  {d.class_name:<12} {d.confidence:.1%}  {[int(v) for v in d.bbox]}")
        if args.output:
            img = cv2.imread(args.image)
            cv2.imwrite(args.output, det.draw(img, r))
            print(f"Saved: {args.output}")

    elif args.camera is not None:
        src = int(args.camera) if args.camera.isdigit() else args.camera
        cap = cv2.VideoCapture(src)
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Animal Detector", det.draw(frame, det.detect(frame)))
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release(); cv2.destroyAllWindows()
    else:
        p.print_help()

if __name__ == "__main__":
    main()
