import pyttsx3
import threading

class OfflineVoice:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)  # Natural speaking speed
        self.lock = threading.Lock()  # Prevents voice overlapping

    def say(self, text):
        def _speak():
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=_speak).start()
