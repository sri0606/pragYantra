import threading
from .utils.aid_hear import live_transcribe, TalkingState

class LiveTranscriber:
    """
    LiveTranscriber class to handle live transcription.

    Methods:
    - __init__(): Initializes the LiveTranscriber object.
    - start(): Starts the live transcription process.
    - terminate(): Terminates the live transcription process.
    """

    def __init__(self):
      self._thread = None
      self._stop_event = threading.Event()
      self._pause_event = threading.Event()
      self._pause_event.clear()  # Initially set the event, so the transcription starts immediately
      self.talking_state = TalkingState()

    def start(self):
      """
      Starts the live transcription process.

      If the transcription is already running, it prints a message and returns.
      Otherwise, it creates a new thread and starts the live_transcribe function.
      """
      if self._thread is not None and self._thread.is_alive():
        print("Transcription is already running")
        return

      self._thread = threading.Thread(target=live_transcribe, 
                      kwargs={'model': 'base', 'non_english': False,'stop_event': self._stop_event,
                              'talking_state':self.talking_state, 'pause_event': self._pause_event,})
      self._thread.start()
      print("Transcribing thread started.")

    def is_voice_detected(self):
        """
        Checks if anyone is talking/ voice activity detection (VAD).

        Returns:
            bool: True if vad, False otherwise.
        """
        return self.talking_state.is_someone_talking()
    
    def get_pause_event(self):
        """
        Returns the pause event.

        Returns:
            threading.Event: The pause event.
        """
        return self._pause_event
    
    def pause(self):
        """
        Pause transcription.
        """
        self._pause_event.clear()

    def resume(self):
        """
        Resume transcription.
        """
        self._pause_event.set()

    def terminate(self):
      """
      Terminates the live transcription process.

      If the transcription is not running, it prints a message and returns.
      Otherwise, it signals the transcription thread to stop, waits for it to finish,
      and resets the stop event so that transcription can be started again.
      """
      if self._thread is None or not self._thread.is_alive():
        print("Transcriber is not running")
        return

      # Signal the transcription thread to stop
      self._stop_event.set()

      # Wait for the transcription thread to finish
      self._thread.join(timeout=1.0)

      # Reset the stop event so we can start the transcription again
      self._stop_event.clear()
      print("Hearing thread stopped.")


if __name__ == "__main__":
    transcriber = LiveTranscriber()
    transcriber.start()
    print("Transcriber started.")
