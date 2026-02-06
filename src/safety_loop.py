import cv2
from ultralytics import YOLO

class SafetyPipeline:
    def __init__(self):
        print("[SYSTEM] Loading YOLOv8 Nano model... (First run downloads 6MB)")
        # 'yolov8n.pt' is the smallest, fastest version. 
        # It will auto-download to your current folder the first time you run it.
        self.model = YOLO("yolov8n.pt")
        
        # We don't need a manual label map anymore; YOLO has one built-in.
        print("[SYSTEM] YOLOv8 Model Loaded.")

    def scan_frame(self, frame):
        if frame is None: return []

        # YOLO handles resizing and normalization internally.
        # We just pass the raw frame.
        # conf=0.40 means "Only show if 40% sure"
        results = self.model(frame, verbose=False, conf=0.40)
        
        found = []
        
        # Parse results to match your existing main.py format
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 1. Get the Label
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                # 2. Get the Confidence Score
                score = float(box.conf[0])
                
                # 3. Get Coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Format for main.py: {'label': 'person', 'score': 85, 'box': [x, y, x, y]}
                found.append({
                    "label": label,
                    "score": int(score * 100),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })

        return found