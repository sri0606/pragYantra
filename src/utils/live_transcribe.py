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

####  inspired from https://github.com/davabase/whisper_real_time   #####
def live_transcribe(model="medium", non_english=False, energy_threshold=1000, 
                    record_timeout=2, phrase_timeout=3, default_microphone='pulse',stop_event=None):
    """
    Transcribes audio from the microphone in real time.

    Parameters:
    model (str): The model to use for transcription. Default is "medium".
    non_english (bool): Whether to use a non-English model. Default is False.
    energy_threshold (int): The energy level threshold for the speech recognizer. Default is 1000.
    record_timeout (int): The maximum length of a single phrase in seconds. Default is 2.
    phrase_timeout (int): The maximum time to wait for a new phrase in seconds. Default is 3.
    default_microphone (str): The default microphone to use. Default is 'pulse'.
    stop_event (threading.Event): An event that will be set when the transcription should stop.
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
    audio_model = whisper.load_model(os.path.abspath(f"models/{model}.pt"))

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

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    # Assume transcript_dir_path is the directory where you want to save the transcripts
    transcript_dir_path = "memory_stream/hearing_logs/"

    # Initialize a timer
    next_save_time = datetime.now() + timedelta(seconds=3)
    text = ''
    
    transcriptions = {}

    while True:
        # Check the stop event at the start of each loop iteration
        if stop_event is not None and stop_event.is_set():
            break
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
                    transcriptions[current_hour] = [{now.strftime("%H%M%S"): text}]
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
