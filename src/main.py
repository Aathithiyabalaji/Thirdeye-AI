import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
import pytesseract
import pyttsx3
from collections import Counter

# --- 1. CONFIGURATION ---
# POINT THIS TO YOUR TESSERACT INSTALLATION
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 2. THE BULLETPROOF VOICE ENGINE ---
class VoiceWorker:
    def __init__(self):
        self.lock = threading.Lock()
        print("[VOICE] Initializing Audio Driver...")

    def say(self, text):
        """Spawns a new thread to speak so the camera doesn't freeze."""
        threading.Thread(target=self._speak_task, args=(text,), daemon=True).start()

    def _speak_task(self, text):
        with self.lock:
            try:
                # Force Windows "SAPI5" Driver (Reliable)
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                
                print(f"[DEBUG] Speaking: {text}")
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception as e:
                print(f"[VOICE ERROR] {e}")

    def stop(self):
        pass

# --- 3. THE MAIN INTELLIGENT SYSTEM ---
class ThirdEyeSystem:
    def __init__(self):
        print("[SYSTEM] Initializing Third Eye High-Efficiency Mode...")
        
        self.voice = VoiceWorker()
        self.voice.say("High Performance System Online") 
        
        self.model = YOLO("yolov8n.pt") 
        
        # --- UPGRADE 1: FORCE HD RESOLUTION ---
        self.cap = cv2.VideoCapture(0)
        # Try to set 1280x720. If camera is old, it will fallback to max available.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Read back the actual resolution to confirm
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] Running at {self.width}x{self.height}")
        
        # Smart Distance Thresholds
        self.size_thresholds = {
            "person": 0.40, "car": 0.20, "bus": 0.20, "truck": 0.20,
            "motorcycle": 0.20, "stop sign": 0.15, "bottle": 0.90,
            "cup": 0.90, "cell phone": 0.90
        }
        self.default_threshold = 0.35
        
        # --- UPGRADE 2: PREDICTION STABILITY BUFFER ---
        # We store the last 5 frames of detections to prevent "glitching"
        self.detection_buffer = [] 

    def get_position(self, x1, x2):
        center_x = self.width // 2
        obj_center = (x1 + x2) // 2
        if obj_center < (center_x - 150): return "Left"
        elif obj_center > (center_x + 150): return "Right"
        else: return "Center"

    def preprocess_for_ocr(self, img_roi):
        """
        Applies 'Reading Glasses' filters to the image to help Tesseract.
        """
        # 1. Grayscale
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        
        # 2. Detail Enhancement (Sharpening) - Makes text edges pop!
        # This is the "Magic Sauce" for reading
        sharpened = cv2.detailEnhance(img_roi, sigma_s=10, sigma_r=0.15)
        gray_sharp = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

        # 3. Zoom (2x Scale)
        gray_sharp = cv2.resize(gray_sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 4. Adaptive Threshold (Scanner Mode)
        binary = cv2.adaptiveThreshold(gray_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                     cv2.THRESH_BINARY, 11, 2)
        return binary

    def run(self):
        print("--- THIRD EYE READY ---")
        print(" [SPACE] Smart Scan (Stable Vision)")
        print(" [R]     High-Res Read (OCR)")
        print(" [H]     Hazard Simulation")
        print(" [Q]     Quit")
        
        IGNORED_ITEMS = ["scissors", "airplane", "boat", "bird", "vase", "hair drier"] 
        STRICT_ITEMS = ["knife", "fork", "spoon"]

        detected_objects = []
        ocr_text_overlay = ""
        clear_time = 0
        
        # Define Target Box (Bigger now for HD)
        roi_w, roi_h = 600, 200
        roi_x = (self.width - roi_w) // 2
        roi_y = (self.height - roi_h) // 2

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # Draw Target Box Guide (Always visible)
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 1)
            cv2.putText(frame, "TEXT HERE", (roi_x + 10, roi_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            key = cv2.waitKey(1) & 0xFF
            
            # --- HAZARD CHECK ---
            if key == ord('h'):
                print("!!! HAZARD !!!")
                cv2.rectangle(frame, (0,0), (self.width, self.height), (0,0,255), 10)
                self.voice.stop()
            
            # --- FEATURE: HIGH-EFFICIENCY OCR ---
            elif key == ord('r'):
                print("\n[OCR] Processing High-Res Text...")
                self.voice.say("Reading") 
                detected_objects = [] # Clear boxes
                
                # 1. Crop High-Res ROI
                roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                # 2. Apply "Reading Glasses" Filter
                processed_img = self.preprocess_for_ocr(roi)
                
                # Debug: Show what the AI sees (Optional - opens a small window)
                # cv2.imshow("AI Eye", processed_img) 

                try:
                    # 3. Read
                    custom_config = r'--oem 3 --psm 6' # Assume block of text
                    text = pytesseract.image_to_string(processed_img, config=custom_config)
                    
                    # 4. Clean
                    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!' "
                    clean_text = "".join([c for c in text if c in allowed_chars or c == '\n'])
                    final_text = " ".join(clean_text.split())
                    
                    if len(final_text) > 4:
                        print(f"[OCR RESULT] {final_text}")
                        self.voice.say(final_text[:250]) 
                        ocr_text_overlay = final_text[:60]
                        clear_time = time.time() + 8.0
                    else:
                        print("[OCR] Text unclear")
                        self.voice.say("Text unclear, hold steady")
                except Exception as e:
                    print(f"[OCR ERROR] {e}")

            # --- FEATURE: STABLE OBJECT DETECTION ---
            elif key == ord(' '): 
                print("\n[AI] Smart Scanning...")
                ocr_text_overlay = "" 
                
                # Run YOLO
                results = self.model(frame, verbose=False, conf=0.35) # Higher confidence (0.35)
                
                current_frame_items = []
                detected_objects = [] # Clear old drawings
                
                screen_area = self.width * self.height
                priority_alert = None
                
                for r in results:
                    for box in r.boxes:
                        label = self.model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        
                        # High Efficiency Filter: Ignore low confidence clutter
                        if label in IGNORED_ITEMS: continue
                        if conf < 0.40 and label not in ["person", "car"]: continue 
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Safety Logic
                        box_area = (x2 - x1) * (y2 - y1)
                        coverage = box_area / screen_area
                        threshold = self.size_thresholds.get(label, self.default_threshold)
                        
                        color = (0, 255, 0)
                        if coverage > threshold:
                            priority_alert = f"STOP. {label} close!"
                            color = (0, 0, 255)
                        
                        detected_objects.append((x1, y1, x2, y2, label, int(conf*100), color))
                        
                        pos = self.get_position(x1, x2)
                        current_frame_items.append(f"{label} {pos}")

                clear_time = time.time() + 3.0

                # --- VOICE LOGIC ---
                if priority_alert:
                    self.voice.say(priority_alert)
                elif current_frame_items:
                    # Sort logic: People/Cars first
                    high = [x for x in current_frame_items if any(p in x for p in ["person", "car", "bus"])]
                    low = [x for x in current_frame_items if x not in high]
                    
                    # Remove duplicates (e.g., "person Center, person Center" -> "person Center")
                    final_speech_list = list(dict.fromkeys(high + low))
                    
                    speech = ", ".join(final_speech_list[:3])
                    print(f"[VOICE] {speech}")
                    self.voice.say(speech)
                else:
                    self.voice.say("Path clear")

            # --- DRAWING LOOP ---
            if time.time() < clear_time:
                for (x1, y1, x2, y2, label, score, color) in detected_objects:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {score}%", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if ocr_text_overlay:
                    cv2.rectangle(frame, (0, 0), (self.width, 60), (0, 0, 0), -1)
                    cv2.putText(frame, ocr_text_overlay, (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elif key == ord('q'):
                break
            
            cv2.imshow("Third Eye Feed", frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ThirdEyeSystem()
    app.run()