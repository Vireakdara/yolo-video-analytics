import time
import cv2
import yaml
from src.inference.detector import Detector
from src.tracking.tracker import Tracker
import requests
import threading
import redis
import json

redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)

def load_config(config_path: str = "./configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def post_detections(detections: list, fps: float = 0.0) -> None:
    try:
        payload = {
            "timestamp": time.time(),
            "fps": round(fps, 1),
            "detections": detections
        }
        redis_client.lpush("detections", json.dumps(payload))
        redis_client.ltrim("detections", 0, 99)
    except Exception as e:
        print(f"Redis error: {e}")

def main():

    config = load_config()
    model_path = config["model"]["path"]
    conf_threshold = config["model"].get("conf_threshold", 0.25)
    video_path = config["video"]["path"]  

    tracker = Tracker(model_path=model_path, conf_threshold=conf_threshold)
    cap = cv2.VideoCapture(video_path)
    redis_client.delete("detections")
    redis_client.set("session", "new")
    print("Redis cleared - starting fresh session")
    class_names = config.get("classes", {})
    
    if not cap.isOpened():
        print("Error: Cannot open video")
        return
    
    frame_count = 0
    fps_start = time.time()
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = tracker.track(frame)

        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        for obj in results:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            track_id = obj["track_id"]
            class_id = obj["class_id"]
            label = f"ID:{track_id} {class_names.get(class_id, 'unknown')}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        threading.Thread(
            target=post_detections,
            args=(results, current_fps),
            daemon=True
        ).start()

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()