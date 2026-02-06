import cv2
import time
import threading
from utils.tts_engine import OfflineVoice
from safety_loop import SafetyPipeline

def main():
    print("--- THIRD EYE: VISUAL MODE ---", flush=True)
    voice = OfflineVoice()
    safety = SafetyPipeline()
    cap = cv2.VideoCapture(1) # Try 0 if 1 fails
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    latest_frame = None
    detections = [] # Shared list for main thread to draw
    ai_busy = False
    
    HAZARDS = ["person", "car", "chair", "mobile phone", "bottle", "cup"]

    def ai_thread():
        nonlocal detections, ai_busy
        thread_voice = OfflineVoice() 
        while True:
            if latest_frame is not None and not ai_busy:
                ai_busy = True
                try:
                    # 'results' is now a list of dictionaries with coordinates
                    results = safety.scan_frame(latest_frame)
                    
                    # Update shared variable safely
                    detections = results 

                    if results:
                        print(f"[AI] Found: {[r['label'] for r in results]}", flush=True)
                        
                        # TTS Logic
                        for r in results:
                            if r['label'] in HAZARDS:
                                # Simple throttling: Don't spam if already speaking
                                if not thread_voice.engine.isBusy():
                                    thread_voice.say(f"{r['label']} detected")
                                break 
                except Exception as e:
                    print(f"AI Error: {e}")
                finally:
                    ai_busy = False
            time.sleep(0.03)

    threading.Thread(target=ai_thread, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Update the frame for the AI to see
        latest_frame = frame.copy() 
        
        # 2. Display Frame: DO NOT FLIP - we want "True View"
        display_frame = frame.copy()

        # --- DRAWING LOOP ---
        if detections:
            for det in detections:
                label = f"{det['label']} {det['score']}%"
                xmin, ymin, xmax, ymax = det['box']
                
                # Draw Green Box
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Draw Label Background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(display_frame, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # DEBUG INDICATOR: Blue circle in top-left means "I am running"
        cv2.circle(display_frame, (30, 30), 10, (255, 0, 0), -1) 
        cv2.putText(display_frame, "ACTIVE", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Third Eye Feed", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()