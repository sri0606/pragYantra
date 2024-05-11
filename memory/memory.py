from datetime import datetime, timedelta
import os
import json
from . import MEMORY_STREAM_DIR

class LiveMemory:
    """
    LiveMemory class retrieve data from memory logs.

    """
    def __init__(self, latest_time_threshold=5):
        """
        Initializes a Memory object.

        Args:
            latest_time_threshold (int, optional): The time threshold (in seconds) for considering a memory entry as the latest. Defaults to 5 seconds.
        """
        self.latest_time_threshold = latest_time_threshold

    def get_latest_memory(self):
        """
        Get the latest data from the JSON files.

        Returns:
            dict: The latest data.
        """
        # Get today's date and format it as YYYYMMDD
        now = datetime.now()
        current_date, current_hour = now.strftime("%Y%m%d %H").split()

        # Construct the file path
        hearing_log_path = os.path.join(MEMORY_STREAM_DIR,"hearing_logs", f"{current_date}_transcript.json")
        vision_log_path = os.path.join(MEMORY_STREAM_DIR,"vision_logs", f"{current_date}_vision.json")

        # Load the data from the file
        try:
            with open(hearing_log_path, 'r') as f:
                heard_data = json.load(f)

                last_heard_time = datetime.strptime(heard_data[current_hour][-1][0],"%H%M%S")
                # Change the date of last_heard_time to the current date
                last_heard_time = last_heard_time.replace(year=now.year, month=now.month, day=now.day)
                # Check if the last heard time is within the latest time threshold
                if now - last_heard_time > timedelta(seconds=self.latest_time_threshold):
                    latest_heard_data = ""
                    recent_heard_data = ""
                else:
                    latest_heard_data = heard_data[current_hour][-1][1]
                    recent_heard_data = heard_data[current_hour][-2][1] if len(heard_data[current_hour]) > 1 else ""
        except Exception as e:
            # print(e)
            latest_heard_data = ""
            recent_heard_data = ""

        try:
            with open(vision_log_path, 'r') as f:
                seen_data = json.load(f)

                last_seen_time = datetime.strptime(seen_data[current_hour][-1][0],"%H%M%S")
                # Change the date of last_heard_time to the current date
                last_seen_time = last_seen_time.replace(year=now.year, month=now.month, day=now.day)
                # Check if the last seen time is within the lastest time threshold
                if now - last_seen_time > timedelta(seconds=self.latest_time_threshold):
                    latest_seen_data = ""
                    recent_seen_data = ""
                else:
                    latest_seen_data = str(seen_data[current_hour][-1][1])
                    recent_seen_data = str(seen_data[current_hour][-2][1]) if len(seen_data[current_hour]) > 1 else ""
        except Exception as e:
            # print(e)
            latest_seen_data = ""
            recent_seen_data = ""
            
        return {"latest":{"vision":latest_seen_data, "audio":latest_heard_data},
                "recent":{"vision":recent_seen_data, "audio":recent_heard_data}}




if __name__ == "__main__":
    memory = LiveMemory(latest_time_threshold=600)
    print(memory.get_latest_memory())