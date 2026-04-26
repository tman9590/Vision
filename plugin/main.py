#!/usr/bin/env python3
"""
plugin/main.py — Scrypted NVR ObjectDetection + Settings plugin.
Bridges AnimalDetector inference engine with Scrypted's camera pipeline.
"""
from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from inference import AnimalDetector, InferenceResult
    _DET_OK = True
except Exception as e:
    print(f"⚠  inference import failed: {e}")
    _DET_OK = False

try:
    import numpy as np
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    from scrypted_sdk import (ScryptedDeviceBase, ObjectDetection,
                               Settings, MediaObject)
    _SDK = True
except ImportError:
    _SDK = False
    class ScryptedDeviceBase:
        def __init__(self): self.storage = {}
    class ObjectDetection: pass
    class Settings: pass

CONFIG_PATH = ROOT / "config.json"

class AnimalDetectionPlugin(ScryptedDeviceBase, ObjectDetection, Settings):

    def __init__(self, nativeId=None):
        super().__init__()
        with open(CONFIG_PATH) as f:
            self.cfg = json.load(f)
        self._detector: Optional[AnimalDetector] = None
        self._load_detector()

    # ── Settings ──────────────────────────────────────────────────────────────

    def _get(self, key):
        return self.storage.get(key, self.cfg["scrypted"]["settings"][key]["default"])

    async def getSettings(self):
        return [{"key":k,"title":v["title"],"description":v.get("description",""),
                 "type":v["type"],"value":self._get(k),
                 **({} if "choices" not in v else {"choices":v["choices"]})}
                for k,v in self.cfg["scrypted"]["settings"].items()]

    async def putSetting(self, key, value):
        self.storage[key] = value
        if key in ("confidence_threshold","iou_threshold","compute_units"):
            self._load_detector()

    # ── Detector ──────────────────────────────────────────────────────────────

    def _load_detector(self):
        if not _DET_OK: return
        try:
            self._detector = AnimalDetector(
                conf_threshold = float(self._get("confidence_threshold")),
                iou_threshold  = float(self._get("iou_threshold")),
                compute_units  = str(self._get("compute_units")),
            )
        except Exception as e:
            print(f"✗ Detector load failed: {e}")
            self._detector = None

    # ── ObjectDetection interface ─────────────────────────────────────────────

    async def getDetectionModel(self):
        return {"name": self.cfg["model"]["name"], "classes": self.cfg["classes"],
                "inputSize": self.cfg["model"]["input_size"]}

    async def detectObjects(self, mediaObject, session=None):
        if not self._detector:
            return {"detections":[], "timestamp": int(time.time()*1000)}
        try:
            frame  = await self._decode(mediaObject)
            result = self._detector.detect(frame)
            return self._to_scrypted(result)
        except Exception as e:
            print(f"⚠  detectObjects error: {e}")
            return {"detections":[], "timestamp": int(time.time()*1000)}

    async def _decode(self, mo):
        if not _CV2_OK: raise RuntimeError("OpenCV not available")
        raw  = await mo.getData()
        arr  = np.frombuffer(raw, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Failed to decode frame")
        return img

    def _to_scrypted(self, result: InferenceResult):
        enabled  = set(self._get("enabled_classes") or [])
        min_area = float(self._get("min_box_area") or 0)
        dets = []
        for d in result.detections:
            if enabled and d.class_name not in enabled: continue
            x1,y1,x2,y2 = d.bbox
            if (x2-x1)*(y2-y1) < min_area: continue
            dets.append(d.to_scrypted())
        return {"detections":dets, "timestamp": int(time.time()*1000), "running":True}


def create_scrypted_plugin():
    return AnimalDetectionPlugin()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    a = p.parse_args()
    plugin = AnimalDetectionPlugin()
    if plugin._detector and _CV2_OK:
        import cv2 as _cv
        frame  = _cv.imread(a.image)
        result = plugin._detector.detect(frame)
        print(json.dumps(result.to_dict(), indent=2))
