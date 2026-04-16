# YOLO Video Analytics Pipeline

A production-grade real-time video analytics system that detects and tracks objects in video streams, pushes live results to a dashboard via WebSocket, and monitors pipeline performance with Prometheus and Grafana.

---

## Architecture

```
Video → Tracker → Redis → API → WebSocket → Dashboard
```

Each video frame is processed by YOLOv8 with ByteTrack for persistent object IDs. Results are pushed into Redis as a fast intermediate buffer. The FastAPI backend reads from Redis and streams updates to the browser dashboard via WebSocket in real time.

---

## Features

- Real-time object detection and tracking with persistent IDs (ByteTrack)
- Redis-backed detection buffer decoupling tracker from API
- Live WebSocket dashboard — no polling
- Session accumulator tracking all unique objects seen throughout the video
- FPS counter measuring actual pipeline throughput
- Dynamic class badges showing all detected object types
- Manual session clear button
- Prometheus metrics endpoint for production monitoring
- Grafana dashboard for FPS, detection rate, and class breakdowns
- ONNX export and benchmarking for inference optimization
- COCO class labels loaded from YAML config

---

## Benchmark

Tested on Intel i5-12450H | NVIDIA RTX 4050 (CPU inference)

| Runtime | FPS | ms/frame |
|---------|-----|----------|
| PyTorch | 13.0 | 76.8ms |
| ONNX Runtime | 20.1 | 49.8ms |
| **Speedup** | **1.54x** | **35% faster** |

> Note: Real pipeline FPS with ByteTrack and Redis overhead runs at ~16 FPS with PyTorch.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Vireakdara/yolo-video-analytics.git
cd yolo-video-analytics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Edit `configs/config.yaml` with your model and video path:

```yaml
model:
  path: "yolov8n.pt"
  conf_threshold: 0.25

video:
  path: "path/to/your/video.mp4"
```

### 3. Start Redis

```bash
redis-server
```

### 4. Start the API

```bash
python src/api/app.py
```

### 5. Run the tracker

```bash
python main.py
```

### 6. Open dashboard

```
http://127.0.0.1:8080/dashboard
```

### 7. View metrics

```
http://127.0.0.1:8080/metrics
```

---

## Project Structure

```
yolo-video-analytics/
├── configs/
│   └── config.yaml          # Model path, video path, class labels
├── src/
│   ├── api/
│   │   └── app.py           # FastAPI backend, WebSocket, Redis, Prometheus
│   ├── dashboard/
│   │   └── ui.html          # Live browser dashboard
│   ├── inference/
│   │   ├── detector.py      # YOLOv8 detection wrapper
│   │   └── benchmark.py     # ONNX vs PyTorch benchmark
│   └── tracking/
│       └── tracker.py       # ByteTrack tracking wrapper
├── main.py                  # Main pipeline: video → tracker → Redis
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| YOLOv8 (Ultralytics) | Object detection model |
| ByteTrack | Persistent multi-object tracking across frames |
| Redis | In-memory queue decoupling tracker from API |
| FastAPI | REST API and WebSocket backend |
| WebSocket | Real-time push from server to dashboard |
| OpenCV | Video frame reading and annotation |
| Prometheus | Metrics scraping and time-series storage |
| Grafana | Monitoring dashboard for FPS and detection rates |
| ONNX Runtime | Optimized inference — 1.54x faster than PyTorch |
| Docker | Containerized deployment |

---

## Results

| Metric | Value |
|--------|-------|
| Pipeline FPS | ~16 FPS (PyTorch + ByteTrack) |
| Inference latency | 76.8ms per frame |
| ONNX speedup | 1.54x over PyTorch (CPU) |
| Objects tracked simultaneously | Up to 12 per session |
| WebSocket update interval | 100ms |