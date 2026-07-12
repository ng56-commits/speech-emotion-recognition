import librosa
import numpy as np


def extract_mfcc(audio, sr, n_mfcc=40):
    """
    Extracts MFCC (vocal texture/timbre) features from audio.
    Returns a fixed-size vector of 40 numbers, averaged across time.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def extract_chroma(audio, sr):
    """
    Extracts Chroma (pitch class) features from audio.
    Returns a fixed-size vector of 12 numbers, averaged across time.
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return chroma.mean(axis=1)