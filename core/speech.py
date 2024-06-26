import threading
from .utils.aid_speech import Pyttsx3Speech,ElevenLabsSpeech, FacebookMMS
import queue
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import time
from datetime import datetime
from . import MEMORY_STREAM_DIR

class LiveSpeech:
    """
    LiveSpeech class to handle live speech.

    Attributes:
        _speaker (Speech): The speech synthesis engine.
        _thread (Thread): The thread for processing speech.
        _queue (Queue): The queue to store speech texts.
        _stop_event (Event): The event to signal the speech processing thread to stop.
    """

    def __init__(self,speaker_model=None):
        """
        Initializes the LiveSpeech class.

        This method initializes the Speech class for speech synthesis,
        initializing the thread, queue, and stop event for managing speech playback.

        Parameters:
            speaker_model (str): The speech synthesis engine to use. Default is None. 
                            Current Options are "pyttsx3", "11labs", and "facebook-mms".

        Returns:
            None
        """
        pygame.mixer.init()
        if speaker_model is None or speaker_model == "pyttsx3":
            self._speaker = Pyttsx3Speech()   

        elif speaker_model == "11labs":
            self._speaker = ElevenLabsSpeech()

        elif speaker_model == "facebook-mms":
            self._speaker = FacebookMMS()

        self._thread = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self.ears_pause_event = None
        self._cleanup_thread = threading.Thread(target=self.cleanup)

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

            save_dir = os.path.join(MEMORY_STREAM_DIR,'audio_logs/')
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
        self._cleanup_thread.start()   

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

    def cleanup(self,sleep_time=20):
        """
        Regularly cleans up the audio logs directory.
        """
        while not self._stop_event.is_set():
            time.sleep(sleep_time)  # Sleep for 20 seconds

            # Delete all files in the directory
            save_dir = os.path.join(MEMORY_STREAM_DIR,'audio_logs/')
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    #if error occured, it most probably means that the file is still being used by speech thread
                    continue
                    # print(f'Failed to delete {file_path}. Reason: {e}')

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

        # Wait for the cleanup thread to finish
        self._cleanup_thread.join(timeout=5.0)  

        # Reset the stop event so we can start the Speech again
        self._stop_event.clear()

        print("Speech thread stopped.")



