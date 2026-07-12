import noisereduce as nr


def reduce_noise(audio_array, sample_rate):
    """
    Reduce background noise from an audio signal.
    """

    reduced_audio = nr.reduce_noise(
        y=audio_array,
        sr=sample_rate
    )

    return reduced_audio, sample_rate


import numpy as np


def trim_or_pad(audio_array, sample_rate, duration=3):
    """
    Trim or pad audio to a fixed duration.
    """

    target_length = duration * sample_rate

    # Trim
    if len(audio_array) > target_length:
        audio_array = audio_array[:target_length]

    # Pad
    elif len(audio_array) < target_length:
        padding = target_length - len(audio_array)

        audio_array = np.pad(
            audio_array,
            (0, padding),
            mode="constant"
        )

    return audio_array, sample_rate