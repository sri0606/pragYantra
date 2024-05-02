import pyttsx3

class Speech:
    """
    Speech class to handle all speech related tasks
    """
    def __init__(self):
        self._speech = None

    def text_to_speech(self, text):
        raise NotImplementedError("Subclass must implement text_to_speech method")


class Pyttsx3Speech(Speech):
    """
    A class that provides text-to-speech functionality using the pyttsx3 library.

    Attributes:
        speech_engine (pyttsx3.Engine): The pyttsx3 speech engine.
        speech_rate (int): The speech rate in words per minute.

    Methods:
        text_to_speech(text): Converts the given text to speech.
        save_to_file(filename, text): Saves the given text to a file.
        play(): Plays the speech.
        set_speech_rate(rate): Sets the speech rate to the specified value.
        get_speech_rate(): Returns the current speech rate.
        set_speech_voice(voice): Sets the speech voice to the specified value.
        get_speech_voices(): Returns the available speech voices.
    """

    def __init__(self):
        super().__init__()
        self.speech_engine = pyttsx3.init()
        self.speech_rate=200

    def speak_instant(self,text:str):
        self.speech_engine.speak(text)

    def text_to_speech(self, text:str):
        # Implementation for pyttsx3 goes here
        self.speech_engine.say(text)
    
    def save_to_file(self, filename:str, text:str):
        self.speech_engine.save_to_file(text, filename)

    def play(self):
        self.speech_engine.runAndWait()

    def set_speech_rate(self,rate):
        self.speech_rate=rate
        self.speech_engine.setProperty('rate', rate)

    def get_speech_rate(self):
        return self.speech_rate
    
    def set_speech_voice(self, voice_id):
        """
        Set the voice for the speech engine.

        :param voice_id: The ID of the voice to use.
        """
        self.speech_engine.setProperty('voice', voice_id)


    def get_speech_voices(self):
        """
        Get ids of all voices available on your machine.

        :return: A dictionary where the keys are the indices and the values are the voice objects.
        """
        voices = {}
        for i, voice in enumerate(self.speech_engine.getProperty('voices')):
            voices[i] = voice.id
        return voices

    def set_volume(self, volume):
        """
        Set the volume of the speech engine. only between 0 and 1

        :param volume: The volume to set.
        """
        if not 0 <= volume <= 1:
            raise ValueError("Volume must be between 0 and 1")

        self.speech_engine.setProperty('volume', volume)

    def get_volume(self):
        """
        Get the current volume of the speech engine.

        :return: The current volume.
        """
        return self.speech_engine.getProperty('volume')

import requests
import os
from dotenv import load_dotenv
load_dotenv()

class ElevenLabsSpeech:
    

    def text_to_speech_eleven_labs_api(self, text):
        """
        This method converts text to audio
        """
        speech_key = os.getenv("SPEECH_KEY")
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
