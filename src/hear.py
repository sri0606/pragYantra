import threading
from utils.aid_hear import live_transcribe

class LiveTranscriber:
    """
    LiveTranscriber class to handle live transcription
    """
    def __init__(self):
      self._thread = None
      self._stop_event = threading.Event()

    def start(self):
      if self._thread is not None and self._thread.is_alive():
          print("Transcription is already running")
          return

      self._thread = threading.Thread(target=live_transcribe, 
                        kwargs={'model': 'base', 'non_english': False,'stop_event': self._stop_event})
      self._thread.start()
      print("Transcribing thread started.")

    def terminate(self):
      if self._thread is None or not self._thread.is_alive():
          print("Transcription is not running")
          return

      # Signal the transcription thread to stop
      self._stop_event.set()

      # Wait for the transcription thread to finish
      self._thread.join()

      # Reset the stop event so we can start the transcription again
      self._stop_event.clear()
      print("Transcribing thread stopped.")
