import cv2
import numpy as np
import os
from ai_edge_litert.interpreter import Interpreter

class SafetyPipeline:
    def __init__(self, model_path='models/detect.tflite', label_path='models/labelmap.txt'):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        f_model = os.path.join(base_dir, 'models', 'detect.tflite')
        f_label = os.path.join(base_dir, 'models', 'labelmap.txt')

        self.interpreter = Interpreter(model_path=f_model, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.h, self.w = self.input_details[0]['shape'][1], self.input_details[0]['shape'][2]

        # TENSOR FINDER (Keep this, it's working)
        self.box_idx, self.class_idx, self.score_idx = -1, -1, -1
        for detail in self.output_details:
            idx = detail['index']
            shape = detail['shape']
            if len(shape) == 3 and shape[-1] == 4: self.box_idx = idx
            elif len(shape) == 2:
                if self.class_idx == -1: self.class_idx = idx
                else: self.score_idx = idx
        
        # Fallbacks
        if self.box_idx == -1: self.box_idx = self.output_details[0]['index']
        if self.class_idx == -1: self.class_idx = self.output_details[1]['index']
        if self.score_idx == -1: self.score_idx = self.output_details[2]['index']

        # --- THE FIX: RAW LOADING ---
        # Do NOT insert 'background'. Trust the raw file.
        with open(f_label, 'r') as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]
        
        # DEBUG: Tell us what Index 0 actually is now
        print(f"[SYSTEM] Index 0 is now: '{self.labels[0]}'")
        print(f"[SYSTEM] Index 1 is now: '{self.labels[1]}'")

    def scan_frame(self, frame):
        if frame is None: return []

        # 1. ASPECT RATIO FIX (Letterboxing)
        old_h, old_w = frame.shape[:2]
        ratio = min(self.w / old_w, self.h / old_h)
        new_w, new_h = int(old_w * ratio), int(old_h * ratio)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        y_offset, x_offset = (self.h - new_h) // 2, (self.w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0) # UINT8

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        classes = self.interpreter.get_tensor(self.class_idx)[0]
        scores = self.interpreter.get_tensor(self.score_idx)[0]

        # Normalize Scores if needed
        if np.max(scores) > 1.1: scores = scores / 255.0

        found = []
        for i in range(len(scores)):
            # LOWER THRESHOLD to 35% to catch the person in bad light
            if scores[i] > 0.35: 
                idx = int(classes[i])
                if idx < len(self.labels):
                    label = self.labels[idx]
                    # We removed the 'background' check because Index 0 might be Person!
                    if label != "???":
                        # Return ID, Label, and Score for total transparency
                        found.append(f"ID:{idx} {label} ({int(scores[i]*100)}%)")
        return found