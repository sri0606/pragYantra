# from whisper_live.client import TranscriptionClient
# client = TranscriptionClient(
#   "localhost",
#   9090,
#   lang="en",
#   translate=False,
#   model="base",
#   use_vad=False,
# )

# client()

import threading
from utils.live_transcribe import live_transcribe

class LiveTranscriber:
    """
    LiveTranscriber class to handle live transcription
    """
    def __init__(self):
      self.thread = None
      self.stop_event = threading.Event()

    def start(self):
      if self.thread is not None and self.thread.is_alive():
          print("Transcription is already running")
          return

      self.thread = threading.Thread(target=live_transcribe, 
                        kwargs={'model': 'base', 'non_english': False,'stop_event': self.stop_event})
      self.thread.start()
      print("Transcribing thread started.")

    def stop(self):
      if self.thread is None or not self.thread.is_alive():
          print("Transcription is not running")
          return

      # Signal the transcription thread to stop
      self.stop_event.set()

      # Wait for the transcription thread to finish
      self.thread.join()

      # Reset the stop event so we can start the transcription again
      self.stop_event.clear()
      print("Transcribing thread stopped.")
