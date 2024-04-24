import requests
import os
from dotenv import load_dotenv
load_dotenv()

speech_key = os.getenv("SPEECH_KEY")

class Speech:
    """
    Speech class to handle all speech related tasks
    """
    def __init__(self):
        self._speech = None

    def text_to_audio(self, text):
        """
        This method converts text to audio
        """
        CHUNK_SIZE = 1024
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": speech_key
        }

        data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        with open('output.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
