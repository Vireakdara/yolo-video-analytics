import cv2
import yaml
from src.inference.detector import Detector
from src.tracking.tracker import Tracker

def load_config(config_path: str = "./configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():

    config = load_config()
    model_path = config["model"]["path"]
    conf_threshold = config["model"].get("conf_threshold", 0.25)
    video_path = config["video"]["path"]  

    tracker = Tracker(model_path=model_path, conf_threshold=conf_threshold)
    cap = cv2.VideoCapture(video_path)
    class_names = config.get("classes", {})
    
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = tracker.track(frame)

        for obj in results:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            track_id = obj["track_id"]
            class_id = obj["class_id"]
            label = f"ID:{track_id} {class_names.get(class_id, 'unknown')}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()