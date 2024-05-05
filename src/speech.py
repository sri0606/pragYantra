import threading
from utils.aid_speech import Pyttsx3Speech
import queue
import os
import pygame
import time
from datetime import datetime

class LiveSpeech:
    """
    LiveSpeech class to handle live speech.

    Attributes:
        _speaker (Speech): The speech synthesis engine.
        _thread (Thread): The thread for processing speech.
        _queue (Queue): The queue to store speech texts.
        _stop_event (Event): The event to signal the speech processing thread to stop.
    """

    def __init__(self):
        """
        Initializes the Speech class.

        This method initializes the Speech class by initializing the Pygame mixer,
        creating an instance of the Pyttsx3Speech class for speech synthesis,
        initializing the thread, queue, and stop event for managing speech playback.

        Parameters:
            None

        Returns:
            None
        """
        pygame.mixer.init()
        self._speaker = Pyttsx3Speech()
        self._thread = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self.ears_pause_event = None

    def set_ears_pause_event(self, event: threading.Event):
        """
        Set the ears pause event.

        Parameters:
            event (Event): The event to pause the ears.

        Returns:
            None
        """
        self.ears_pause_event = event

    def _process_speech(self):
        """
        Process the speech by retrieving text from the queue and saving it to an audio file.

        This method runs in a loop until the stop event is set. It retrieves text from the queue,
        saves it to an audio file, and plays the audio file.

        Returns:
            None
        """
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            save_dir = os.path.join(os.getcwd(), "memory_stream/audio_logs/")
            filename = save_dir + datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"

            # Save the speech to an audio file in a separate thread
            self._speaker.save_to_file(filename, text)
            self._play_audio(filename)

    def _play_audio(self, filename):
        """
        Plays the audio file specified by the given filename.

        Args:
            filename (str): The path to the audio file to be played.

        Returns:
            None
        """
        pygame.mixer.music.load(filename)

        #pauses hearing ability so that own speech is not transcribed
        if self.ears_pause_event is not None:
            self.ears_pause_event.set()

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if self._stop_event.is_set():
                pygame.mixer.music.stop()
                break

            time.sleep(0.1)
        
        #resumes hearing ability
        if self._stop_event is not None:
            self.ears_pause_event.clear() 

    def start(self):
        """
        Starts the speech processing thread.

        If the speech processing thread is already running, it prints a message and returns.

        Returns:
            None
        """
        if self._thread is not None and self._thread.is_alive():
            print("Speech is already running")
            return

        self._thread = threading.Thread(target=self._process_speech)
        self._thread.start()

        print("Speech thread started")

    def is_speaking(self):
        """
        Checks if the speech is running.

        Returns:
            bool: True if the speech is running, False otherwise.
        """
        return pygame.mixer.music.get_busy()
    
    def speak(self, text):
        """
        Adds the given text to the speech queue.

        Args:
            text (str): The text to be spoken.

        Returns:
            None
        """
        # self.shut_up()
        self._queue.put(text)

    def pause(self):
        """
        Pauses the audio playback.

        Returns:
            None
        """
        pygame.mixer.music.pause()

    def shut_up(self):
        """
        Stops the audio playback.

        Returns:
            None
        """
        pygame.mixer.music.stop()

    def continue_speaking(self):
        """
        Resumes the audio playback.

        Returns:
            None
        """
        pygame.mixer.music.unpause()

    def terminate(self):
        """
        Terminates the speech processing thread.

        If the speech processing thread is not running, it returns.

        Returns:
            None
        """
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



