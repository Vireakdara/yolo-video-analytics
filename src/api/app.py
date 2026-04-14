from fastapi import FastAPI
from typing import List, Dict, Any
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="YOLO Video Analytics API")
@app.get("/dashboard")
def dashboard():
    return FileResponse("E:\\yolo-video-analytics\\src\\dashboard\\ui.html")

latest_detections: List[Dict[str, Any]] = []

@app.post("/")
def root():
    return {"message": "YOLO Video Analytics API is running"}

@app.get("/detections")
def get_detections():
    return {
        "count": len(latest_detections),
        "detections": latest_detections
        }

@app.post("/detections")
def update_detections(detections: List[Dict[str, Any]]):
    global latest_detections
    latest_detections = detections
    return {"message": "Detections updated successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)