import librosa
import numpy as np


def preprocess_audio(audio_arr, sample_rate, target_sr=16000):
    if sample_rate != target_sr:
        audio_arr = librosa.resample(audio_arr.astype(float), orig_sr=sample_rate, target_sr=target_sr)
    audio_arr = audio_arr / np.max(np.abs(audio_arr))
    return audio_arr.astype(np.float32)
