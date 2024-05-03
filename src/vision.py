import threading
from utils.vision_aid import VisionAid

class LiveVision:
    """
    Vision class to handle all vision related tasks
    """
    def __init__(self):
        """
        Constructor
        """
        self._eyes = VisionAid(stop_event_wait_time=5, save_to_json_interval=3)
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        """
        Starts the vision thread if it is not already running.
        """
        if self.thread is not None and self.thread.is_alive():
          print("Vision is already running")
          return

        self.thread = threading.Thread(target=self._eyes.get_visual_context, args=(self.stop_event,))
        self.thread.start()

        print("Vision thread started")


    def stop(self):
        """
        Stops the vision thread if it is running.
        """
        if self.thread is None or not self.thread.is_alive():
            return

        # Signal the Vision thread to stop
        self.stop_event.set()

        # Wait for the Vision thread to finish
        self.thread.join()

        # Reset the stop event so we can start the Vision again
        self.stop_event.clear()

        print("Vision thread stopped.")






