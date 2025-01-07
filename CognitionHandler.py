import numpy as np
import whisper

from model_handlers.HybridModelHandler import HybridModelHandler
from tools import AudioTools


class CognitionHandler:
    def __init__(self):
        self.whisper: whisper.Whisper = whisper.load_model("")
        self.hmh = HybridModelHandler()

    def transcribe_audio(self, audio_arr: np.ndarray, sample_rate) -> str:
        try:
            audio_arr = AudioTools.preprocess_audio(audio_arr, sample_rate)
            transcription = self.whisper.transcribe(audio_arr)["text"]
            return transcription

        except Exception as e:
            print(f"Error in transcribe_audio: {e}")

    def query(self, query: str):
        return self.hmh.query(query)
