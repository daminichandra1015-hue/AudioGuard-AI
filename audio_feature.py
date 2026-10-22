
import librosa
import numpy as np


def extract_features(audio_path):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    features = np.mean(mfcc, axis=1)
    duration = len(audio) / sample_rate

    return features, mfcc, duration, sample_rate
