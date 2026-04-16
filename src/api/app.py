import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis
import uvicorn
from fastapi.responses import FileResponse
import yaml
import time
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response


app = FastAPI(title="YOLO Video Analytics API")

# Metrics
fps_gauge = Gauge("yolo_fps", "Current inference FPS")
objects_gauge = Gauge("yolo_objects_tracked", "Current number of objects being ")
detections_counter = Counter("yolo_detections_total", "Total detections since start")
class_counter = Counter("yolo_class_detections_total", "Detections per class", ["class_name"])
latency_histogram = Histogram("yolo_inference_latency_ms", "Inference latency in ms")


redis_client = redis.Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    decode_responses=True
)


def load_config():
    with open('configs\\config.yaml', "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

@app.get("/")
def root():
    return {"message": "YOLO Video Analytics API is running"}


@app.get("/classes")
def get_classes():
    return config.get("classes", {})


@app.get("/detections")
def get_detections():
    data = redis_client.lindex("detections", 0)

    if data is None:
        return {"count": 0, "fps": 0, "detections": []}

    payload = json.loads(data) # type: ignore
    detections = payload.get("detections", [])

    return {
        "count": len(detections),
        "fps": payload.get("fps", 0),
        "timestamp": payload.get("timestamp"),
        "detections": detections
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_objects = {}  # track_id -> latest detection data

    try:
        while True:
            session = redis_client.get("session")
            if session == "new":
                session_objects = {}  # reset on new session
                await websocket.send_text("CLEAR")
                redis_client.delete("session")

            data = redis_client.lindex("detections", 0)
            if data:
                payload = json.loads(data)
                detections = payload.get("detections", [])

                fps_gauge.set(payload.get("fps", 0))
                objects_gauge.set(len(detections))
                detections_counter.inc(len(detections))

                for det in detections:
                    class_name = str(det.get("class_id", "unknown"))
                    class_counter.labels(class_name=class_name).inc()

                # Merge into session accumulator
                for det in detections:
                    session_objects[det["track_id"]] = det

                await websocket.send_text(
                    json.dumps({
                        "fps": payload.get("fps", 0),
                        "detections": list(session_objects.values())
                    })
                )
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/dashboard")
def dashboard():
    return FileResponse('src\\dashboard\\ui.html')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)