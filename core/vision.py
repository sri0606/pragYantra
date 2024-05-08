import threading
from .utils.aid_vision import VisionAid

class LiveVision:
    """
    Vision class to handle all vision related tasks
    """
    def __init__(self):
        """
        Constructor
        """
        self._eyes = VisionAid(stop_event_wait_time=5, save_to_json_interval=3)
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """
        Starts the vision thread if it is not already running.
        """
        if self._thread is not None and self._thread.is_alive():
          print("Vision is already running")
          return
        self._eyes.set_blind(False)
        self._thread = threading.Thread(target=self._eyes.get_visual_context, args=(self._stop_event,))
        self._thread.start()

        print("Vision thread started")


    def terminate(self):
        """
        Stops the vision thread if it is running.
        """
        if self._thread is None or not self._thread.is_alive():
            return
        self._eyes.set_blind(True)
        # Signal the Vision thread to stop
        self._stop_event.set()

        # Wait for the Vision thread to finish
        self._thread.join()

        # Reset the stop event so we can start the Vision again
        self._stop_event.clear()
        
        print("Vision thread stopped.")






