import queue
import threading

import pyttsx3


class TextToSpeech:
    """Thread-safe, non-blocking wrapper around pyttsx3.

    Uses a single background worker thread and a queue to avoid
    'run loop already started' errors from multiple concurrent runAndWait() calls.
    """

    def __init__(self) -> None:
        self.engine = pyttsx3.init()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()

    def speak_async(self, text: str) -> None:
        """Enqueue text to be spoken by the background worker."""
        if not text:
            return
        self._queue.put(text)
