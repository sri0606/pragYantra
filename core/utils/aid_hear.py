#! python3.7
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import json
from .. import MODELS_DIR, MEMORY_STREAM_DIR

class TalkingState:
    """
    State of talking
    """
    def __init__(self):
        #check if someone is talking
        self.is_talking = False
        # self.id_person_talking = None

    def set_is_talking(self, is_talking,person_id=None):
        """
        Set the state of the talking.

        Parameters:
        is_talking (bool): The state of the talking.
        person_id (int): The id of the person talking.
        """
        self.is_talking = is_talking

    def is_someone_talking(self):
        """
        Check if someone is talking.

        Returns:
        bool: True if someone is talking, False otherwise.
        """
        return self.is_talking

####  inspired from https://github.com/davabase/whisper_real_time   #####
def live_transcribe(talking_state: TalkingState,model="medium", non_english=False, energy_threshold=1000, 
                    record_timeout=2, phrase_timeout=3, default_microphone='pulse',
                    stop_event=None,pause_event=None,):
    """
    Transcribes audio from the microphone in real time.

    Parameters:
    talking_state (TalkingState): The state of the talking.
    model (str): The model to use for transcription. Default is "medium".
    non_english (bool): Whether to use a non-English model. Default is False.
    energy_threshold (int): The energy level threshold for the speech recognizer. Default is 1000.
    record_timeout (int): The maximum length of a single phrase in seconds. Default is 2.
    phrase_timeout (int): The maximum time to wait for a new phrase in seconds. Default is 3.
    default_microphone (str): The default microphone to use. Default is 'pulse'.
    stop_event (threading.Event): An event that will be set when the transcription should stop.
    pause_event (threading.Event): An event that will be set when the transcription should pause, typically when LiveSpeech is talking.
    """

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    if model != "large" and not non_english:
            model = model + ".en"
    try:
        #try loading from project's local models dir
        audio_model = whisper.load_model(os.path.join(MODELS_DIR, f"{model}.pt"))
    except FileNotFoundError:
        audio_model = whisper.load_model(model)

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

        # # Convert the audio data to an array of integers
        # audio_int = np.frombuffer(data, dtype=np.int16)

        # ### rough implementation of live VAD
        # # Calculate the energy (volume) of the audio data
        # energy = np.sum(audio_int**2)

        # # If the energy is above the threshold, someone is talking
        # if energy > recorder.energy_threshold:
        #     talking_state.set_is_talking(True)
        # else:
        #     talking_state.set_is_talking(False)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Whisper Model loaded.\n")

    transcript_dir_path = os.path.join(MEMORY_STREAM_DIR, "hearing_logs")

    # Initialize a timer
    next_save_time = datetime.now() + timedelta(seconds=3)
    text = ''
    
    transcriptions = {}

    while True:
        # Check the stop event at the start of each loop iteration
        if stop_event is not None and stop_event.is_set():
            break
        # Check the pause event at the start of each loop iteration
        if pause_event is not None and pause_event.is_set():
            continue
        try:
            now = datetime.now()
            current_hour = now.strftime("%H")

            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text += result['text'].strip()

                if current_hour not in transcriptions:
                    transcriptions[current_hour] = []

                if phrase_complete:
                    transcriptions[current_hour] = [(now.strftime("%H%M%S"), text)]
                    text = ''

                if now >= next_save_time:
                    date_string = now.strftime("%Y%m%d")
                    filename = f"{date_string}_transcript.json"
                    filepath = os.path.join(transcript_dir_path, filename)

                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            if f.read().strip():
                                f.seek(0)  # reset file pointer to beginning
                                existing_data = json.load(f)
                            else:
                                existing_data = {}
                    else:
                        existing_data = {}

                    if current_hour not in existing_data:
                        existing_data[current_hour] = []

                    existing_data[current_hour].extend(transcriptions[current_hour])

                    with open(filepath, 'w') as f:
                        json.dump(existing_data, f)

                    transcriptions[current_hour].clear()

                    next_save_time = now + timedelta(seconds=3)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    # talking_state = TalkingState()
    # live_transcribe(talking_state, model='base',non_english=False)
    print(MODELS_DIR)