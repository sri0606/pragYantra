import threading
from utils.aid_speech import Pyttsx3Speech
import queue
import os
import pygame
import time
from datetime import datetime

class LiveSpeech:
    """
    LiveSpeech class to handle live speech
    """
    def __init__(self):
        pygame.mixer.init()
        self._speaker = Pyttsx3Speech()
        self._thread = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()

    def _process_speech(self):
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            save_dir = os.path.join(os.getcwd(), "memory_stream/audio_logs/")
            filename=save_dir+datetime.now().strftime("%Y%m%d%H%M%S")+".wav"

            # Save the speech to an audio file in a separate thread
            self._speaker.save_to_file(filename, text)
            self._play_audio(filename)

    def _play_audio(self, filename):
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if self._stop_event.is_set():
                pygame.mixer.music.stop()
                break

            time.sleep(0.1)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
          print("Speech is already running")
          return
        
        self._thread = threading.Thread(target=self._process_speech)
        self._thread.start()

        print("Speech thread started")

    def speak(self,text):
        self.shut_up()
        self._queue.put(text)

    def pause(self):
        # Pause the audio playback
        pygame.mixer.music.pause()

    def shut_up(self):
        # Stop the audio playback
        pygame.mixer.music.stop()

    def continue_speaking(self):
        # Resume the audio playback
        pygame.mixer.music.unpause()


    def terminate(self):
        if self._thread is None or not self._thread.is_alive():
            return
        # Stop the audio playback and the speech processing thread
        pygame.mixer.music.stop()

        # Signal the Speech thread to stop
        self._stop_event.set()

        # Wait for the Speech thread to finish
        self._thread.join(timeout=1.0)

        # Reset the stop event so we can start the Speech again
        self._stop_event.clear()

        print("Speech thread stopped.")



