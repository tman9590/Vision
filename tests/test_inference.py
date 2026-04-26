#!/usr/bin/env python3
"""
tests/test_inference.py — 24 unit tests (no model file required).

Run:
    python -m pytest tests/ -v
"""
import json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from inference import (Detection, InferenceResult, Preprocessor,
                        Postprocessor, nms, xywh2xyxy)


@pytest.fixture
def cfg():
    with open(ROOT / "config.json") as f:
        return json.load(f)

@pytest.fixture
def classes(cfg): return cfg["classes"]
@pytest.fixture
def pre():         return Preprocessor(640)
@pytest.fixture
def post(classes): return Postprocessor(classes, 0.40, 0.45)
@pytest.fixture
def bgr():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)


class TestConfig:
    def test_required_keys(self, cfg):
        for k in ("model","classes","class_colors","training","export","scrypted"):
            assert k in cfg

    def test_class_count(self, cfg):        assert len(cfg["classes"]) == 20
    def test_no_duplicates(self, cfg):      assert len(cfg["classes"]) == len(set(cfg["classes"]))
    def test_input_size(self, cfg):         assert cfg["model"]["input_size"] == [640, 640]
    def test_conf_range(self, cfg):         assert 0 < cfg["model"]["confidence_threshold"] < 1
    def test_iou_range(self, cfg):          assert 0 < cfg["model"]["iou_threshold"] < 1

    def test_colors_match_classes(self, cfg):
        missing = [c for c in cfg["classes"] if c not in cfg["class_colors"]]
        assert missing == [], f"Missing colors: {missing}"


class TestPreprocessor:
    def test_shape_landscape(self, pre, bgr):
        blob, _, _ = pre(bgr)
        assert blob.shape == (1, 3, 640, 640)

    def test_dtype(self, pre, bgr):
        blob, _, _ = pre(bgr)
        assert blob.dtype == np.float32

    def test_normalised(self, pre, bgr):
        blob, _, _ = pre(bgr)
        assert blob.min() >= 0.0 and blob.max() <= 1.0

    def test_square_no_pad(self, pre):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, scale, (pw, ph) = pre(img)
        assert scale == pytest.approx(1.0, abs=0.01)
        assert pw == 0 and ph == 0

    def test_portrait(self, pre):
        img = np.zeros((1280, 720, 3), dtype=np.uint8)
        blob, scale, _ = pre(img)
        assert blob.shape == (1, 3, 640, 640)
        assert scale == pytest.approx(0.5, abs=0.01)


class TestBoxUtils:
    def test_xywh2xyxy(self):
        b = np.array([[100., 100., 50., 50.]])
        r = xywh2xyxy(b)
        np.testing.assert_allclose(r, [[75., 75., 125., 125.]])

    def test_nms_removes_overlap(self):
        boxes  = np.array([[0,0,100,100],[5,5,105,105]], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep   = nms(boxes, scores, 0.5)
        assert len(keep) == 1 and keep[0] == 0

    def test_nms_keeps_separate(self):
        boxes  = np.array([[0,0,50,50],[200,200,250,250]], dtype=np.float32)
        scores = np.array([0.9, 0.85])
        assert len(nms(boxes, scores, 0.5)) == 2

    def test_nms_empty(self):
        assert nms(np.empty((0,4),dtype=np.float32), np.empty(0), 0.5) == []


class TestPostprocessor:
    def _raw(self, nc=20, n=8400):
        rng = np.random.default_rng(7)
        raw = rng.random((1, 4+nc, n), dtype=np.float32) * 0.3
        raw[0, 0, 0] = 320.0   # cx
        raw[0, 1, 0] = 320.0   # cy
        raw[0, 2, 0] = 100.0   # w
        raw[0, 3, 0] = 100.0   # h
        raw[0, 4, 0] = 0.95    # bear confidence
        return raw

    def test_finds_bear(self, post):
        dets = post(self._raw(), 640, 640, 1.0, (0,0))
        assert any(d.class_name == "bear" for d in dets)

    def test_conf_threshold(self, classes):
        pp   = Postprocessor(classes, 0.99, 0.45)
        raw  = np.random.default_rng(1).random((1,24,8400), dtype=np.float32)*0.5
        assert pp(raw, 640, 640, 1.0, (0,0)) == []

    def test_bbox_in_bounds(self, post):
        for d in post(self._raw(), 640, 640, 1.0, (0,0)):
            x1,y1,x2,y2 = d.bbox
            assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 640

    def test_norm_range(self, post):
        for d in post(self._raw(), 640, 640, 1.0, (0,0)):
            assert all(0. <= v <= 1. for v in d.bbox_norm)


class TestDetection:
    def _d(self):
        return Detection("dog", 5, 0.876, (10.,20.,110.,120.), (.01,.02,.11,.12))

    def test_dict_keys(self):
        assert {"class","class_id","confidence","bbox","bbox_norm"} <= self._d().to_dict().keys()

    def test_scrypted_keys(self):
        s = self._d().to_scrypted()
        assert {"className","score","boundingBox"} <= s.keys()
        assert {"x","y","width","height"} <= s["boundingBox"].keys()

    def test_scrypted_dims(self):
        bb = self._d().to_scrypted()["boundingBox"]
        assert bb["width"]  == pytest.approx(100.)
        assert bb["height"] == pytest.approx(100.)


class TestInferenceResult:
    def test_animals_present_dedup(self):
        dets = [Detection("cat",2,0.9,(0,0,50,50),(0,0,.5,.5)),
                Detection("cat",2,0.7,(10,10,60,60),(0,0,.5,.5)),
                Detection("dog",5,0.8,(100,100,200,200),(0,0,.5,.5))]
        r = InferenceResult(detections=dets, latency_ms=5., image_w=640, image_h=480)
        assert sorted(r.animals_present) == ["cat","dog"]

    def test_dict_structure(self):
        r = InferenceResult(latency_ms=5., image_w=1280, image_h=720)
        assert {"detections","animals_present","count","latency_ms"} <= r.to_dict().keys()
