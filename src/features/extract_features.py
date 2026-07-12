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



def extract_features_partB(audio, sr):
    """
    Extract:
    - Spectral Contrast (7)
    - Zero Crossing Rate (1)
    - RMS Energy (1)
    - Spectral Centroid (1)

    Returns:
        numpy array of shape (10,)
    """

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )
    spectral_contrast = np.mean(
        spectral_contrast,
        axis=1
    )

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # RMS Energy
    rms = np.mean(
        librosa.feature.rms(y=audio)
    )

    # Spectral Centroid
    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    features = np.hstack([
        spectral_contrast,
        zcr,
        rms,
        centroid
    ])

    return features



if __name__ == "__main__":
    # Replace with the path to one of your audio files
    
    file_path = r"C:\Users\varsh\OneDrive\Desktop\ml_project\speech-emotion-recognition\data\raw\ravdess\Actor_01\03-01-01-01-01-01-01.wav"

    audio, sr = librosa.load(file_path, sr=None)

    features = extract_features_partB(audio, sr)

    print("Features:")
    print(features)
    print("Shape:", features.shape)