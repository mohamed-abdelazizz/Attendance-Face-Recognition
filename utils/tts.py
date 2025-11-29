import pyttsx3
import queue
import threading


class TextToSpeech:
    """Async text-to-speech using pyttsx3 with a background thread."""

    def __init__(self) -> None:
        # Initialize pyttsx3 engine
        self.engine = pyttsx3.init()
        # Queue to store texts that need to be spoken
        self._queue = queue.Queue()
        # Start a background thread to speak texts
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        """Background thread: read texts from queue and speak them one by one."""
        while True:
            text = self._queue.get()  # wait for a text to speak
            if text is None:  # special signal to stop thread (not used here)
                break
            self.engine.say(text)
            self.engine.runAndWait()  # block here until speaking finishes

    def speak_async(self, text: str) -> None:
        """Add text to the queue to speak in background."""
        if text:  # ignore empty text
            self._queue.put(text)
