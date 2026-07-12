import os
import librosa

# Emotion mapping for RAVDESS
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def load_ravdess(dataset_path):
    audio_data = []
    labels = []
    sample_rates = []

    for actor in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                path = os.path.join(actor_path, file)

                audio, sr = librosa.load(path, sr=None)

                emotion_code = file.split("-")[2]
                emotion = RAVDESS_EMOTIONS[emotion_code]

                audio_data.append(audio)
                labels.append(emotion)
                sample_rates.append(sr)

    return audio_data, labels, sample_rates


def load_tess(dataset_path):
    audio_data = []
    labels = []
    sample_rates = []

    for emotion_folder in os.listdir(dataset_path):

        folder_path = os.path.join(dataset_path, emotion_folder)

        if not os.path.isdir(folder_path):
            continue

        # OAF_angry -> angry
        emotion = emotion_folder.split("_")[-1].lower()

        # TESS uses "pleasant_surprise"
        if emotion == "surprise":
            emotion = "surprised"

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                path = os.path.join(folder_path, file)

                audio, sr = librosa.load(path, sr=None)

                audio_data.append(audio)
                labels.append(emotion)
                sample_rates.append(sr)

    return audio_data, labels, sample_rates


def load_all_datasets(ravdess_path, tess_path):

    rav_audio, rav_labels, rav_sr = load_ravdess(ravdess_path)

    tess_audio, tess_labels, tess_sr = load_tess(tess_path)

    audio = rav_audio + tess_audio
    labels = rav_labels + tess_labels
    sample_rates = rav_sr + tess_sr

    return audio, labels, sample_rates





#FOR TESTING PURPOSES
if __name__ == "__main__":

    ravdess_path = "../../data/raw/ravdess"
    tess_path = "../../data/raw/tess"

    audio, labels, sample_rates = load_all_datasets(
        ravdess_path,
        tess_path,
    )

    print("Total audio files:", len(audio))
    print("First label:", labels[0])
    print("Shape:", audio[0].shape)
    print("Sample rate:", sample_rates[0])