from __future__ import annotations
from typing import Any, Dict, List  
from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path: str, conf_threshold: float = 0.25) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track(self, frame: Any) -> List[Dict[str, Any]]:
        results = self.model.track(
            frame, 
            verbose=False, 
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml"
        )
        tracked: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            for box, track_id in zip(result.boxes, result.boxes.id): # type: ignore
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tracked.append({
                    "track_id": int(track_id.item()),
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(box.cls[0].item()),
                    "confidence": float(box.conf[0].item()),
                })
        return tracked