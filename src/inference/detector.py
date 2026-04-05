from __future__ import annotations
from typing import Any, Dict, List
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25)->None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> List[Dict[str, Any]]:
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        detections: List[Dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "confidence": confidence,
                })
        return detections
            
