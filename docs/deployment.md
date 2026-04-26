# Scrypted Deployment Guide

## Requirements

- Scrypted ≥ 0.3.0 on macOS 13+ with Apple Silicon
- Python 3.10+ in Scrypted's runtime
- `model/AnimalDetector.mlpackage` present (or use ONNX fallback on any OS)

## Install Plugin

1. Open Scrypted → **Settings → Plugins → Install from Local Path**
2. Select this project directory
3. Scrypted installs Python dependencies automatically

## Assign to Cameras

1. Open camera → **Extensions → Object Detection**
2. Select **Animal Detection (CoreML)**
3. Save

## Configure

| Setting | Recommended | Notes |
|---|---|---|
| Confidence | 0.40 | Lower for wildlife cams, raise to cut false positives |
| Compute Units | cpuAndNeuralEngine | Best performance on Apple Silicon |
| Enabled Classes | (empty = all) | Set specific animals to reduce noise |
| Min Box Area | 400 | Ignores detections smaller than ~20×20 px |

## Performance

| Compute Units | Latency | Power draw |
|---|---|---|
| `cpuOnly` | ~40 ms | Low |
| `cpuAndGPU` | ~15 ms | Medium |
| `cpuAndNeuralEngine` | ~8 ms | Very low |

## Troubleshooting

**No detections** — lower confidence to 0.25, verify `.mlpackage` is in `model/`

**High false positives** — raise confidence to 0.55+, restrict enabled classes

**High CPU** — switch to `cpuAndNeuralEngine`, lower camera analysis frame rate

**ONNX fallback on Linux** — expected; install coremltools on macOS for Neural Engine
