import time
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
ONNX_PATH = "yolov8n.onnx"
TEST_RUNS = 100

def export_to_onnx():
    print("Exporting YOLOv8 to ONNX...")
    model = YOLO(MODEL_PATH)
    model.export(format="onnx", simplify=True)
    print(f"Exported to {ONNX_PATH}")

def benchmark_pytorch(frame):
    model = YOLO(MODEL_PATH)

    # warmup
    for _ in range(10):
        model(frame, verbose=False)

    start = time.time()
    for _ in range(TEST_RUNS):
        model(frame, verbose=False)
    elapsed = time.time() - start
    print(f"PyTorch YOLOv8n: {TEST_RUNS/elapsed:.2f} FPS")

    fps = TEST_RUNS / elapsed
    ms = (elapsed / TEST_RUNS) * 1000
    return fps, ms

def benchmark_onnx(frame):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] 
    session = ort.InferenceSession(ONNX_PATH, providers=providers)

    img = cv2.resize(frame, (640, 640))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img,0).astype(np.float32) / 255.0
    input_name = session.get_inputs()[0].name

    # warmup
    for _ in range(10):
        session.run(None, {input_name: img})

    start = time.time()
    for _ in range(TEST_RUNS):
        session.run(None, {input_name: img})
    elapsed = time.time() - start

    fps = TEST_RUNS / elapsed
    ms = (elapsed / TEST_RUNS) * 1000
    return fps, ms

def main():
    # export if not exists
    import os
    if not os.path.exists(ONNX_PATH):
        export_to_onnx()

    # create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("\nRunning benchmarks...")
    print("-" * 40)

    pytorch_fps, pytorch_ms = benchmark_pytorch(frame)
    print(f"PyTorch:      {pytorch_fps:.1f} FPS | {pytorch_ms:.1f}ms per frame")

    onnx_fps, onnx_ms = benchmark_onnx(frame)
    print(f"ONNX Runtime: {onnx_fps:.1f} FPS | {onnx_ms:.1f}ms per frame")

    speedup = onnx_fps / pytorch_fps
    print("-" * 40)
    print(f"Speedup: {speedup:.2f}x faster with ONNX")

if __name__ == "__main__":
    main()