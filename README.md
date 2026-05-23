# thermal_mot

# 🔥 Thermal-MOT: Real-Time Multi-Object Tracking with Edge AI Optimization

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![C++](https://img.shields.io/badge/C++-17-00599C?style=flat-square&logo=cplusplus)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx)
![TensorRT](https://img.shields.io/badge/TensorRT-8.x-76B900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> Real-time multi-object tracking on FLIR thermal imagery — combining signal-domain preprocessing, deep learning detection, algorithm-level tracker benchmarking, and edge-optimized C++/CUDA deployment.

Developed as a graduation thesis at **Ankara University, EEE Department**.  
Internship context: **ASELSAN Sivas Precision Optics — Night and Thermal Vision Systems.**

---

## Technical Scope

| Domain | Implementation |
|---|---|
| **Signal Processing** | FFT-based noise profiling, AGC, CLAHE, crossover zone contrast analysis |
| **Machine Learning** | YOLO11 fine-tuned on FLIR thermal dataset (PyTorch + ONNX) |
| **Algorithms** | Tracker benchmark — ByteTrack, DeepSORT, BoT-SORT, Hybrid 4+4+3 |
| **Edge AI / CUDA** | TensorRT INT8/FP16, CUDA-accelerated preprocessing, multithreaded pipeline |
| **C++ Inference** | TensorRT C++ API engine, pipelined I/O with producer-consumer threading |
| **Performance** | Latency profiling, memory budget analysis, FPS benchmarking across backends |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Thermal-MOT Pipeline                           │
│                                                                      │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐   │
│  │  FLIR Input │──▶│ Signal Preproc.  │──▶│  YOLO11 Detector   │   │
│  │ (.seq/.mp4) │   │ (CUDA kernels)   │   │  Python / C++ API  │   │
│  └─────────────┘   └──────────────────┘   └─────────┬──────────┘   │
│                                                      │               │
│  ┌───────────────────────────────────────────────────▼────────────┐ │
│  │                      Tracker Layer (Python)                    │ │
│  │     ByteTrack │ DeepSORT │ BoT-SORT │ Hybrid 4+4+3            │ │
│  └───────────────────────────────────────────────────┬────────────┘ │
│                                                      │               │
│  ┌───────────────────────────────────────────────────▼────────────┐ │
│  │           CrossoverDetector + CrossoverPredictor               │ │
│  │  Thermal Kalman │ STMC │ Triple-Threshold │ ReID Suppression   │ │
│  └───────────────────────────────────────────────────┬────────────┘ │
│                                                      │               │
│  ┌───────────────────────────────────────────────────▼────────────┐ │
│  │           C++ / TensorRT Deployment (edge_cpp/)                │ │
│  │   Multithreaded pipeline │ CUDA streams │ Memory pool mgmt    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
thermal-mot/
│
├── signal_processing/
│   ├── thermal_normalizer.py        # CLAHE, histogram eq., AGC
│   ├── noise_analysis.py            # FFT-based thermal noise profiling
│   ├── contrast_detector.py         # Crossover zone detection (spatial contrast)
│   └── notebooks/
│       └── signal_analysis.ipynb
│
├── detection/
│   ├── yolo_inference.py            # YOLO11 PyTorch wrapper
│   ├── onnx_export.py               # Export + ONNX validation
│   └── configs/
│       └── yolo11_thermal.yaml
│
├── tracking/
│   ├── bytetrack_wrapper.py
│   ├── deepsort_wrapper.py
│   ├── botsort_wrapper.py
│   ├── hybrid_tracker.py            # 4+4+3 hybrid architecture
│   ├── crossover_detector.py
│   ├── crossover_predictor.py
│   └── modules/
│       ├── thermal_kalman.py        # Thermal-Aware Kalman Covariance
│       ├── stmc.py                  # Spatial-Temporal Motion Consistency
│       ├── triple_threshold.py      # Triple-Threshold IoU Association
│       └── reid_suppressor.py       # Crossover-Aware ReID Suppression
│
├── edge_ai/
│   ├── quantize_int8.py             # PTQ INT8 calibration
│   ├── tensorrt_build.py            # TensorRT engine builder (Python)
│   ├── latency_profiler.py          # Per-layer latency + memory profiling
│   └── benchmark.py                 # FPS benchmark: PyTorch vs ONNX vs TRT
│
├── edge_cpp/                        # ★ C++ inference engine
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp                 # Entry point
│   │   ├── trt_engine.cpp           # TensorRT C++ API wrapper
│   │   ├── trt_engine.hpp
│   │   ├── cuda_preprocess.cu       # CUDA kernel: resize + normalize
│   │   ├── pipeline.cpp             # Multithreaded producer-consumer pipeline
│   │   └── memory_pool.cpp          # GPU memory pool manager
│   └── README.md                    # Build instructions
│
├── evaluation/
│   ├── tracker_comparison.py        # MOTA / MOTP / IDF1 metrics
│   ├── crossover_eval.py
│   └── figures/
│
├── data/
│   ├── synthetic_crossover.py
│   └── dataset_diagnostic.py
│
├── results/
├── requirements.txt
└── README.md
```

---

## C++ / CUDA Component — `edge_cpp/`

The C++ module implements a **production-grade inference pipeline** using the TensorRT C++ API. Key design decisions:

### Multithreaded Pipeline
```
Thread 1 (Producer):  Camera / file I/O → CUDA preprocess kernel → input buffer
Thread 2 (Inference): TensorRT enqueue → CUDA stream sync
Thread 3 (Consumer):  Post-process + draw → output buffer
```
Decoupled with a **lock-free ring buffer** to minimize inter-thread latency.

### CUDA Preprocessing Kernel
```cpp
// cuda_preprocess.cu — fused resize + normalize on GPU
__global__ void preprocess_kernel(
    const uchar* src, float* dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float scale, float mean, float std
);
```
Eliminates CPU-GPU memory transfer bottleneck in high-FPS scenarios.

### OOP Design
```
TRTEngine          — model load, engine build, inference
CUDAPreprocessor   — CUDA kernel wrapper
MemoryPool         — GPU buffer lifecycle
InferencePipeline  — orchestrates threads + data flow
```

---

## Module Highlights — Python

### Hybrid Tracker — `tracking/hybrid_tracker.py`

**4+4+3 architecture** extends BoT-SORT with thermal-specific modules:

| Module | Function |
|---|---|
| Thermal-Aware Kalman Covariance | Adapts process noise to local thermal contrast |
| Spatial-Temporal Motion Consistency | Rejects physically implausible trajectory candidates |
| Triple-Threshold Association | Low / mid / high IoU gates (vs. binary threshold) |
| Crossover-Aware ReID Suppression | Disables ReID inside active XO zones |

### Signal Processing — `signal_processing/`
```python
from signal_processing.contrast_detector import CrossoverZoneAnalyzer

analyzer = CrossoverZoneAnalyzer(contrast_low=8.0, sharpness_low=50.0)
xo_mask = analyzer.detect(frame_gray)  # binary XO zone mask
```

---

## Performance Benchmarks

### Inference Backend Comparison (NVIDIA RTX 3060)

| Backend | Precision | FPS | Latency (ms) |
|---|---|---|---|
| PyTorch (Python) | FP32 | ~38 | ~26 |
| ONNX Runtime | FP32 | ~61 | ~16 |
| TensorRT Python | FP16 | ~94 | ~11 |
| TensorRT Python | INT8 | ~118 | ~8.5 |
| **TensorRT C++** | **INT8** | **~141** | **~7.1** |

> C++ pipeline with CUDA preprocessing eliminates Python overhead.  
> Jetson Orin results pending.

### Tracker Comparison

| Tracker | MOTA ↑ | MOTP ↑ | IDF1 ↑ | ID Sw. ↓ | XO Recovery ↑ |
|---|---|---|---|---|---|
| ByteTrack | 71.4 | 78.2 | 66.1 | 43 | 51% |
| DeepSORT | 68.9 | 76.5 | 64.8 | 61 | 44% |
| BoT-SORT | 73.1 | 79.4 | 68.3 | 38 | 57% |
| **Hybrid (ours)** | **76.8** | **81.2** | **72.5** | **27** | **74%** |

---

## Installation

### Python
```bash
git clone https://github.com/<your-username>/thermal-mot.git
cd thermal-mot
pip install -r requirements.txt
```

### C++ (edge_cpp)
```bash
cd edge_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DTENSORRT_ROOT=/usr/local/tensorrt \
         -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
make -j$(nproc)
```
**Requirements:** CUDA 12.x, TensorRT 8.x, OpenCV 4.x, CMake 3.18+

---

## Quick Start

```bash
# Python pipeline
python tracking/tracker_comparison.py --video data/sample.mp4 --model detection/weights/best.pt

# Export to ONNX + TensorRT
python detection/onnx_export.py --weights best.pt --imgsz 640
python edge_ai/tensorrt_build.py --onnx model.onnx --precision int8

# Full benchmark
python edge_ai/benchmark.py --backends pytorch onnx tensorrt

# C++ inference
./build/thermal_mot --engine model_int8.engine --input data/sample.mp4
```

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| C++ (OOP, templates, threading) | `edge_cpp/` |
| CUDA programming | `cuda_preprocess.cu` |
| TensorRT (Python + C++ API) | `edge_ai/`, `edge_cpp/src/trt_engine.cpp` |
| ONNX export + validation | `detection/onnx_export.py` |
| Signal analysis & algorithms | `signal_processing/`, `tracking/modules/` |
| Performance profiling | `edge_ai/latency_profiler.py` |
| Data structures & design patterns | Pipeline, pool, observer in `edge_cpp/` |
| Git + modular project structure | This repo |

---

## References

- ByteTrack — Zhang et al., ECCV 2022  
- DeepSORT — Wojke et al., ICIP 2017  
- BoT-SORT — Aharon et al., arXiv 2022  
- YOLO11 — Ultralytics, 2024  
- Hartley & Zisserman — Multiple View Geometry in Computer Vision  
- Szeliski — Computer Vision: Algorithms and Applications, Ch. 7  

---

## License

MIT License. See `LICENSE` for details.
