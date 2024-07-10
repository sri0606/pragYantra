import numpy as np
import sounddevice as sd
import threading
from collections import deque
import time
from datetime import datetime, timedelta
import whisper
import torch
from queue import Queue
import os
import re
from .. import MODELS_DIR

class AudioProcessor:
    def __init__(self, sample_rate=16000, frame_duration=0.5, energy_threshold_db=10, 
                 detection_window=1.0, memory_size=50, buffer_size=30, 
                 model="base", non_english=False, phrase_timeout=3,
                 silence_threshold_db=-30, end_of_speech_timeout=2.0,
                 max_silence_ratio=0.8):
        # AudioMonitor attributes
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.energy_threshold_db = energy_threshold_db
        self.detection_window = detection_window
        self.external_speech_detected = threading.Event()
        self.stop_event = threading.Event()
        self.background_energy = None
        self.energy_memory = deque(maxlen=memory_size)
        
        # Transcription attributes
        self.audio_buffer = np.array([], dtype=np.float32)
        self.memory_buffer = deque(maxlen=buffer_size)
        self.current_transcription = ""
        self.text_chunks = ""
        self.transcribing_event = threading.Event()
        self.data_queue = Queue()
        self.min_audio_length = 2 * self.sample_rate  # Minimum audio length for processing
        self.max_audio_length = 20 * self.sample_rate  # Maximum audio length before forced transcription
        self.max_silence_ratio = max_silence_ratio

        # Phrase completion attributes
        self.phrase_timeout = phrase_timeout
        self.phrase_time = None
        
        # Shared attributes
        self.stream = None
        self._monitor_thread = None
        
        # End of speech detection attributes
        self.end_of_speech_timeout = end_of_speech_timeout
        self.last_speech_time = None
        self.speech_ongoing = False
        self.end_of_speech_detected = threading.Event()
        self.post_speech_wait_time = 2
        self.last_eos_time = None

        # attributes for silence detection
        self.silence_threshold_db = silence_threshold_db
        self.silence_duration = 0

        # Whisper model
        self.model = model
        self.non_english = non_english
        self.audio_model = self._load_whisper_model()

    def _load_whisper_model(self):
        if self.model != "large" and not self.non_english:
            self.model = self.model + ".en"
        try:
            return whisper.load_model(os.path.join(os.getcwd(), "models", f"{self.model}.pt"))
        except FileNotFoundError:
            return whisper.load_model(self.model)

    def start_processing(self):
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self.stop_event.clear()
            self.external_speech_detected.clear()
            self.transcribing_event.set()  # Start in transcribing mode
            self._monitor_thread = threading.Thread(target=self._process_audio)
            self._monitor_thread.start()
        return self._monitor_thread

    def stop_processing(self):
        self.stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _set_speech_detected_properties(self,current_time):
        self.last_speech_time = current_time
        if not self.speech_ongoing:
            self.speech_ongoing = True
            self.end_of_speech_detected.clear()
            self.last_eos_time = None
            self.silence_duration = 0

    def _set_end_of_speech_properties(self,current_time):
        self.last_eos_time = current_time
        self.last_speech_time = self.last_eos_time - self.silence_duration
        if self.speech_ongoing:
            self.speech_ongoing = False
            self.end_of_speech_detected.set()

    def _process_audio(self):
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Error in audio stream: {status}")
            
            audio_np = indata.flatten().astype(np.float32)
            
            # Calculate audio level in dB
            epsilon = 1e-10  # Small value to avoid log(0)
            audio_level_db = 20 * np.log10(np.abs(audio_np).mean() + epsilon)
            
            current_time = time.time()
            if self.transcribing_event.is_set():
                self.data_queue.put(audio_np)
                # Transcribing mode
                if audio_level_db > self.silence_threshold_db:
                    print(f"Speech detected: {audio_level_db:.2f} dB > {self.silence_threshold_db:.2f} dB")
                    self._set_speech_detected_properties(current_time)
                else:
                    self.silence_duration += len(audio_np) / self.sample_rate
                    if not self.last_eos_time and self.silence_duration >= self.end_of_speech_timeout:
                        print(f"End of speech detected after {self.silence_duration:.2f} seconds of silence")
                        self._set_end_of_speech_properties(current_time)
            else:
                # Monitoring mode
                if self.background_energy is None:
                    self.background_energy = audio_level_db
                else:
                    self.background_energy = 0.95 * self.background_energy + 0.05 * audio_level_db

                self.energy_memory.append(audio_level_db)
                avg_energy = sum(self.energy_memory) / len(self.energy_memory)

                if not self.last_speech_time or current_time - self.last_speech_time > self.detection_window:
                    if audio_level_db > self.background_energy + self.energy_threshold_db:
                        print(f"External speech detected! Energy: {audio_level_db:.2f} dB, Avg: {avg_energy:.2f} dB, Background: {self.background_energy:.2f} dB")
                        self.external_speech_detected.set()
                        self._set_speech_detected_properties(current_time)

                    else:
                        self.external_speech_detected.clear()
                        if self.speech_ongoing and (current_time - self.last_speech_time > self.end_of_speech_timeout):
                            self._set_end_of_speech_properties(current_time)
                            print("End of speech detected")

        self.stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=self.sample_rate,
                                     blocksize=self.frame_size)

        with self.stream:
            while not self.stop_event.is_set():
                if self.transcribing_event.is_set():
                    self._process_audio_chunk()
                else:
                    sd.sleep(100)

    def _process_audio_chunk(self):
        
        current_time = time.time()
        # Collect audio data
        # Step 1: Copy Data from Queue
        data_snapshot = []
        while not self.data_queue.empty():
            data_snapshot.append(self.data_queue.get())

        # Step 2: Process Copied Data
        for data in data_snapshot:
            self.audio_buffer = np.concatenate((self.audio_buffer, data))


        # Process if we have enough audio data and speech has ended or buffer is too long
        if self.end_of_speech_detected.is_set() and \
            len(self.audio_buffer) > self.min_audio_length and \
            len(self.audio_buffer) > self.max_audio_length or \
            self.last_eos_time and\
            (current_time-self.last_eos_time)>self.post_speech_wait_time:

            self._transcribe()

    def _transcribe(self):
        print(f"Preparing to transcribe.... {len(self.audio_buffer)/self.sample_rate:.2f} sec")
        
        # if self._is_mostly_silent(self.audio_buffer):
        #     print("Audio buffer is mostly silent. Skipping transcription.")
        #     self.audio_buffer = np.array([], dtype=np.float32)
        #     return

        result = self.audio_model.transcribe(self.audio_buffer, fp16=torch.cuda.is_available())
        transcribed_text = result['text'].strip()
        
        if self.is_valid_text(transcribed_text):
            self.text_chunks += transcribed_text + " "
            self._complete_current_phrase()

        # Clear the audio buffer after transcription
        self.audio_buffer = np.array([], dtype=np.float32)

    @staticmethod
    def is_valid_text(text):
        pattern = r"^(?! *[\., ]*$).+"
        return bool(re.match(pattern, text))
    
    def _is_mostly_silent(self, audio):
        epsilon = 1e-10
        audio_levels = 20 * np.log10(np.abs(audio) + epsilon)
        silent_samples = np.sum(audio_levels <= self.silence_threshold_db)
        silence_ratio = silent_samples / len(audio)
        return silence_ratio > self.max_silence_ratio
    
    def _adjust_silence_threshold(self):
        silence_threshold_adjustment_rate = 0.05
        if len(self.background_silence_levels) > 0:
            avg_silence_level = sum(self.background_silence_levels) / len(self.background_silence_levels)
            # Adjust the silence threshold
            self.silence_threshold_db = (1 - silence_threshold_adjustment_rate) * self.silence_threshold_db + \
                                        silence_threshold_adjustment_rate * (avg_silence_level + 5)  # 5 dB buffer


    def _complete_current_phrase(self):
        completed_phrase = self.text_chunks.strip()
        if completed_phrase:
            self.memory_buffer.append((datetime.now(), completed_phrase))
            self.current_transcription += completed_phrase + " "
            print(f"Phrase completed: {completed_phrase}")
            self.text_chunks = ""

    def is_ready_for_processing(self):
        """
        Returns True if ready for processing transcript.
        If the end of speech is detected and the post-speech wait time has elapsed, return True.
        """
        return self.current_transcription and (time.time()-self.last_eos_time)>self.post_speech_wait_time

    def pause_transcription(self):
        """Pause transcription and start monitoring for external speech"""
        self.transcribing_event.clear()
        self.external_speech_detected.clear()

    def resume_transcription(self):
        """Resume transcription and stop monitoring for external speech"""
        self.transcribing_event.set()
        self.external_speech_detected.clear()
        self.speech_ongoing = False
        self.end_of_speech_detected.clear()
        print("Transcription resumed")

    def force_complete_phrase(self):
        """Force completion of the current phrase"""
        self._complete_current_phrase()

    
    def is_end_of_speech(self):
        """Check if end of speech is detected"""
        return self.end_of_speech_detected.is_set()

    def wait_for_end_of_speech(self, timeout=None):
        """Wait for the end of speech to be detected"""
        return self.end_of_speech_detected.wait(timeout)

    def get_recent_context(self, relevance_threshold=300):
        now = datetime.now()
        recent_transcriptions = [
            text for timestamp, text in self.memory_buffer
            if (now - timestamp).total_seconds() <= relevance_threshold
        ]
        return " ".join(recent_transcriptions)

    def is_external_speech_detected(self):
        return self.external_speech_detected.is_set()

    def clear_audio_data(self):
        with threading.Lock():
            while not self.data_queue.empty():
                self.data_queue.get()
        self.audio_buffer = np.array([], dtype=np.float32)

if __name__ == '__main__':
    # talking_state = TalkingState()
    # live_transcribe(talking_state, model='base',non_english=False)
    print(MODELS_DIR)