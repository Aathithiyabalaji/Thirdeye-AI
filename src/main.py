import cv2
import time
import threading
from utils.tts_engine import OfflineVoice
from safety_loop import SafetyPipeline

def main():
    print("--- THIRD EYE: RAW INDEX MODE ---", flush=True)
    voice = OfflineVoice()
    safety = SafetyPipeline()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    latest_frame = None
    detections = []
    ai_busy = False
    
    # Simple list - ensure 'person' is lowercase
    HAZARDS = ["person", "car", "chair", "mobile phone", "cell phone", "bottle"]

    def ai_thread():
        nonlocal detections, ai_busy
        thread_voice = OfflineVoice() 
        while True:
            if latest_frame is not None and not ai_busy:
                ai_busy = True
                try:
                    res = safety.scan_frame(latest_frame)
                    
                    # Log EVERYTHING so we see the raw IDs
                    if res:
                        print(f"\n[AI] {res}", flush=True)
                        
                        # Check for matches
                        for r in res:
                            # Parse "ID:0 person (88%)" -> "person"
                            label_only = r.split(' ')[1] 
                            if label_only in HAZARDS:
                                # Speak it immediately
                                thread_voice.say(f"{label_only} ahead")
                                break # Only speak one at a time
                    else:
                        print(".", end="", flush=True)
                finally:
                    time.sleep(0.3)
                    ai_busy = False
            time.sleep(0.01)

    threading.Thread(target=ai_thread, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = cv2.flip(frame, 1)
        latest_frame = frame.copy() 
        
        cv2.putText(display_frame, "PI-AI: SCANNING", (10, 20), 1, 1, (0, 255, 0), 1)
        cv2.imshow("Third Eye Feed", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()