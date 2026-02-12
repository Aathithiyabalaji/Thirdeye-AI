import pyttsx3
import threading

class VoiceWorker:
    def __init__(self):
        # Lock ensures two threads don't fight for the speaker
        self.lock = threading.Lock()

    def say(self, text):
        """Spawns a new thread to speak so the camera doesn't freeze."""
        threading.Thread(target=self._speak_task, args=(text,), daemon=True).start()

    def _speak_task(self, text):
        # We use the lock so two voices don't talk over each other
        with self.lock:
            try:
                # 1. Initialize a FRESH engine every time (The Nuclear Fix)
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                
                # 2. Speak
                print(f"[DEBUG] Speaking: {text}")
                engine.say(text)
                engine.runAndWait()
                
                # 3. Kill the engine immediately to prevent freezing
                del engine
            except Exception as e:
                print(f"[VOICE ERROR] {e}")

    def stop(self):
        """
        We can't forcefully kill a running thread easily in Python,
        but since we create a fresh engine every time, 
        the 'stuck' risk is already gone.
        """
        pass