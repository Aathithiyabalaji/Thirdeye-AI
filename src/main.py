import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
import pytesseract
import pyttsx3

# --- 1. CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 2. THE BULLETPROOF VOICE ENGINE ---
class VoiceWorker:
    def __init__(self):
        self.lock = threading.Lock()
        print("[VOICE] Initializing Audio Driver...")

    def say(self, text):
        threading.Thread(target=self._speak_task, args=(text,), daemon=True).start()

    def _speak_task(self, text):
        with self.lock:
            try:
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
        print("[SYSTEM] Initializing Third Eye Ultimate...")
        self.voice = VoiceWorker()
        self.voice.say("System Online") 
        self.model = YOLO("yolov8n.pt") 
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.width = 640
        self.height = 480
        
        self.size_thresholds = {
            "person": 0.40, "car": 0.20, "bus": 0.20, "truck": 0.20,
            "motorcycle": 0.20, "stop sign": 0.15, "bottle": 0.90,
            "cup": 0.90, "cell phone": 0.90
        }
        self.default_threshold = 0.35

    def get_position(self, x1, x2):
        center_x = self.width // 2
        obj_center = (x1 + x2) // 2
        if obj_center < (center_x - 100): return "Left"
        elif obj_center > (center_x + 100): return "Right"
        else: return "Center"

    def run(self):
        print("--- THIRD EYE READY ---")
        print(" [SPACE] Scan Objects")
        print(" [R]     Read Text (Target Box Mode)")
        print(" [H]     Hazard Simulation")
        print(" [Q]     Quit")
        
        IGNORED_ITEMS = ["scissors", "airplane", "boat", "bird", "vase", "hair drier"] 
        STRICT_ITEMS = ["knife", "fork", "spoon"]

        detected_objects = []
        ocr_text_overlay = ""
        clear_time = 0

        # Define the "Target Box" for reading (Center of screen)
        # x, y, width, height
        # Make ROI larger for more pixels (centered)
        roi_w, roi_h = 540, 240
        roi_x = (self.width - roi_w) // 2
        roi_y = (self.height - roi_h) // 2

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # Draw the Target Box Guide (White Rectangle) so you know where to aim
            # We draw this faintly on every frame
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 1)
            cv2.putText(frame, "ALIGN TEXT HERE", (roi_x + 130, roi_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('h'):
                print("!!! HAZARD !!!")
                cv2.rectangle(frame, (0,0), (640,480), (0,0,255), 10)
                cv2.putText(frame, "STOP!", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                self.voice.stop()
            
            # --- FEATURE 2: TARGET BOX READER (OCR) - INDUSTRIAL GRADE ---
            elif key == ord('r'):
                print("\n[OCR] Reading Target Box...")
                self.voice.say("Reading") 
                
                # Clear old detection boxes
                detected_objects = [] 
                
                # Crop the ROI
                roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                # Save original ROI for debug
                cv2.imwrite("debug_1_original.png", roi)

                # --- Deskew function ---
                def deskew(image):
                    coords = np.column_stack(np.where(image > 0))
                    if len(coords) < 10:
                        return image
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle
                    (h, w) = image.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


                # --- PIPELINE 1: Grayscale, upscaled ---
                img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.resize(img_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                img_gray = deskew(img_gray)

                # --- PIPELINE 2: CLAHE on grayscale ---
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img_gray)

                # --- PIPELINE 3: Color, upscaled ---
                img_color = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                # Deskew not applied to color (optional)

                # --- PIPELINE 4: Binarized (Otsu) ---
                _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # --- Show live preview of all candidates (stacked) ---
                preview = np.hstack([
                    cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR),
                    img_color,
                    cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
                ])
                preview = cv2.resize(preview, None, fx=0.5, fy=0.5)
                cv2.imshow("OCR Preview", preview)

                try:
                    best_text = ""
                    best_source = ""
                    # Try a wider range of PSM modes
                    psm_modes = [6, 3, 11, 7, 8, 13]
                    # Add morphological tweaks: original, eroded, dilated
                    kernel = np.ones((2, 2), np.uint8)
                    sources = [
                        ("Gray", img_gray),
                        ("CLAHE", img_clahe),
                        ("Color", img_color),
                        ("Binarized", img_bin),
                        ("Eroded", cv2.erode(img_bin, kernel, iterations=1)),
                        ("Dilated", cv2.dilate(img_bin, kernel, iterations=1))
                    ]
                    for src_name, img in sources:
                        for psm in psm_modes:
                            config = f'-l eng --oem 1 --psm {psm}'
                            text = pytesseract.image_to_string(img, config=config)
                            allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!'-:;$%&()\" "
                            clean_text = "".join([c for c in text if c in allowed_chars or c == '\n'])
                            candidate = " ".join(clean_text.split())
                            print(f"  [{src_name} PSM {psm}] -> '{candidate[:50]}...'")
                            score = len(candidate) + (candidate.count(' ') * 5)
                            best_score = len(best_text) + (best_text.count(' ') * 5)
                            if score > best_score:
                                best_text = candidate
                                best_source = f"{src_name} PSM {psm}"
                            if len(candidate) > 10:
                                break
                    final_text = best_text
                    print(f"  [BEST] {best_source}")
                    # --- Smart post-processing: pick the best line ---
                    def score_line(line):
                        # Score: more uppercase, more digits, longer, fewer non-letters
                        upper = sum(1 for c in line if c.isupper())
                        digits = sum(1 for c in line if c.isdigit())
                        alpha = sum(1 for c in line if c.isalpha())
                        nonalpha = sum(1 for c in line if not c.isalnum() and c != ' ')
                        return upper + digits + len(line) - nonalpha

                    lines = [l.strip() for l in best_text.split('\n') if l.strip()]
                    if lines:
                        best_line = max(lines, key=score_line)
                        print(f"[OCR RESULT] {best_line}")
                        self.voice.say(best_line[:200])
                        ocr_text_overlay = best_line[:60]
                        clear_time = time.time() + 10.0
                    else:
                        print("[OCR] No text found. Try better lighting or move closer.")
                        self.voice.say("No text found. Try moving closer.")
                except Exception as e:
                    print(f"[OCR ERROR] {e}")

            # --- FEATURE 3: SMART DISTANCE (YOLO) ---
            elif key == ord(' '): 
                print("\n[AI] Scanning...")
                # Clear old text when scanning for objects
                ocr_text_overlay = "" 
                
                results = self.model(frame, verbose=False, conf=0.25)
                detected_objects = [] 
                found_text_list = []
                priority_alert = None
                screen_area = self.width * self.height
                
                for r in results:
                    for box in r.boxes:
                        label = self.model.names[int(box.cls[0])]
                        score = int(box.conf[0] * 100)
                        
                        if label in IGNORED_ITEMS: continue
                        if label in STRICT_ITEMS and score < 50: continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_area = (x2 - x1) * (y2 - y1)
                        coverage = box_area / screen_area
                        threshold = self.size_thresholds.get(label, self.default_threshold)
                        
                        if coverage > threshold:
                            priority_alert = f"STOP. {label} very close!"
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                        
                        detected_objects.append((x1, y1, x2, y2, label, score, color))
                        pos = self.get_position(x1, x2)
                        found_text_list.append(f"{label} {pos}")

                clear_time = time.time() + 3.0

                if priority_alert:
                    self.voice.say(priority_alert)
                elif found_text_list:
                    high_priority = ["person", "car", "bus", "truck", "motorcycle"]
                    urgent = [x for x in set(found_text_list) if x.split()[0] in high_priority]
                    normal = [x for x in set(found_text_list) if x.split()[0] not in high_priority]
                    speech = ", ".join((urgent + normal)[:3])
                    print(f"[VOICE] {speech}")
                    self.voice.say(speech)
                else:
                    self.voice.say("Nothing detected")

            # --- DRAWING LOOP ---
            if time.time() < clear_time:
                for (x1, y1, x2, y2, label, score, color) in detected_objects:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {score}%", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if ocr_text_overlay:
                    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
                    cv2.putText(frame, ocr_text_overlay, (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif key == ord('q'):
                break
            
            cv2.imshow("Third Eye Feed", frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ThirdEyeSystem()
    app.run()