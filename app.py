"""
==================================================================================
EchoSense AI — Speech Emotion Recognition
----------------------------------------------------------------------------------
Frontend UI only (Streamlit).

This file implements ONLY the user interface, layout, styling, and structure.
All machine learning logic — feature extraction, preprocessing, model loading,
and prediction — is assumed to already exist elsewhere and is called through
clearly marked placeholder functions below.

Author (UI): Built for Varshitha's B.Tech CSE ML Project
Institution : IIITDM Kurnool
==================================================================================
"""

import streamlit as st
import time
import requests
from datetime import datetime

# ----------------------------------------------------------------------------
# NOTE FOR BACKEND INTEGRATION
# ----------------------------------------------------------------------------
# Replace the bodies of the functions below with your actual backend calls.
# Do NOT change their function signatures unless you also update the calls
# made to them further down in the UI code.
# ----------------------------------------------------------------------------

def extract_features_placeholder(audio_file):
    """
    PLACEHOLDER: Call your existing feature extraction function here.
    Example:
        from feature_extraction import extract_features
        return extract_features(audio_file)
    """
    pass


def load_model_placeholder():
    """
    PLACEHOLDER: Call your existing model loading function here.
    Example:
        from model_utils import load_model
        return load_model("emotion_model.pkl")
    """
    pass


def predict_emotion_placeholder(audio_file):
    """
    Sends audio file to FastAPI backend and returns prediction.
    """

    files = {
        "file": (
            audio_file.name,
            audio_file.getvalue(),
            "audio/wav"
        )
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict/",
            files=files
        )

        if response.status_code == 200:
            result = response.json()

            predicted_emotion = result["predicted_emotion"]

            confidence_dict = result["confidence"]

            confidence = max(confidence_dict.values())

            return predicted_emotion.capitalize(), confidence

        else:
            st.error(response.text)
            return None, None

    except Exception as e:
        st.error(f"API Error: {e}")
        return None, None


# ==================================================================================
# APP CONFIGURATION
# ==================================================================================

st.set_page_config(
    page_title="EchoSense AI | Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Emotion -> emoji mapping used across the UI
EMOTION_EMOJI_MAP = {
    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😠",
    "Neutral": "😐",
    "Fearful": "😨",
    "Disgust": "🤢",
    "Surprised": "😲",
    "Calm": "😌",
}


# ==================================================================================
# CUSTOM CSS — load_css()
# ==================================================================================

def load_css():
    """Injects all custom CSS for the premium dark AI theme."""
    st.markdown(
        """
        <style>
        /* ---------- Global Theme ---------- */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', 'Poppins', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 20% 20%, #1b1035 0%, #0b0f2b 40%, #05060f 100%);
            color: #E6E6F0;
        }

        /* Hide default Streamlit chrome for a cleaner product feel */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ---------- Hero Section ---------- */
        .hero-container {
            text-align: center;
            padding: 50px 20px 30px 20px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(114, 9, 183, 0.25), rgba(0, 180, 216, 0.15));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 40px rgba(114, 9, 183, 0.25);
            margin-bottom: 35px;
            animation: fadeIn 1.2s ease-in-out;
        }

        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00F5FF, #7B2FF7, #F72585);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 6px;
            letter-spacing: 1px;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            font-weight: 500;
            color: #B9C2FF;
            margin-bottom: 14px;
        }

        .hero-description {
            font-size: 1.0rem;
            color: #9FA8DA;
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ---------- Generic Card ---------- */
        .glass-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 24px 26px;
            margin-bottom: 22px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.35);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }

        .glass-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 40px rgba(123, 47, 247, 0.35);
            border-color: rgba(0, 245, 255, 0.4);
        }

        .card-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #00F5FF;
            margin-bottom: 8px;
        }

        .card-text {
            color: #C7CBEB;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        /* ---------- Upload Section ---------- */
        .upload-card {
            border: 2px dashed rgba(0, 245, 255, 0.35);
            border-radius: 22px;
            padding: 30px;
            text-align: center;
            background: linear-gradient(135deg, rgba(0,245,255,0.05), rgba(123,47,247,0.06));
            transition: border-color 0.3s ease, background 0.3s ease;
            margin-bottom: 20px;
        }

        .upload-card:hover {
            border-color: #00F5FF;
            background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(123,47,247,0.1));
        }

        /* ---------- Section Headers ---------- */
        .section-header {
            font-family: 'Poppins', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 30px 0 15px 0;
            padding-left: 14px;
            border-left: 5px solid #00F5FF;
        }

        /* ---------- Predict Button ---------- */
        div.stButton > button {
            width: 100%;
            padding: 16px 0;
            font-size: 1.15rem;
            font-weight: 700;
            font-family: 'Poppins', sans-serif;
            color: white;
            background: linear-gradient(90deg, #7B2FF7, #00F5FF);
            border: none;
            border-radius: 16px;
            box-shadow: 0 0 20px rgba(0, 245, 255, 0.45), 0 0 40px rgba(123, 47, 247, 0.3);
            transition: all 0.3s ease-in-out;
        }

        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(0, 245, 255, 0.75), 0 0 60px rgba(123, 47, 247, 0.55);
            color: white;
        }

        div.stButton > button:active {
            transform: scale(0.98);
        }

        /* ---------- Result Card ---------- */
        .result-card {
            text-align: center;
            padding: 35px;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(123,47,247,0.18), rgba(0,245,255,0.10));
            border: 1px solid rgba(0, 245, 255, 0.35);
            box-shadow: 0 0 40px rgba(123, 47, 247, 0.35);
            margin: 20px 0 30px 0;
            animation: popIn 0.6s ease;
        }

        .result-emoji {
            font-size: 5rem;
            margin-bottom: 10px;
        }

        .result-emotion {
            font-family: 'Poppins', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00F5FF, #F72585);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .result-confidence {
            font-size: 1.1rem;
            color: #C7CBEB;
            margin-top: 6px;
        }

        /* ---------- Visualization Placeholder ---------- */
        .viz-placeholder {
            border: 1px dashed rgba(255,255,255,0.18);
            border-radius: 18px;
            padding: 60px 20px;
            text-align: center;
            color: #7C86B8;
            background: rgba(255,255,255,0.02);
            font-size: 0.95rem;
        }

        /* ---------- Native bordered containers (st.container(border=True)) ---------- */
        /* Used for the upload box and audio preview cards so real widgets (file uploader,
           audio player) sit inside a properly themed box without breaking their HTML. */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
            border: 1px solid rgba(0, 245, 255, 0.25) !important;
            border-radius: 20px !important;
            box-shadow: 0 4px 25px rgba(0,0,0,0.35);
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(0, 245, 255, 0.5) !important;
            box-shadow: 0 8px 30px rgba(123, 47, 247, 0.3);
        }

        /* ---------- Tabs (Upload vs Record) ---------- */
        button[data-baseweb="tab"] {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: #9FA8DA;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00F5FF !important;
        }

        div[data-baseweb="tab-highlight"] {
            background-color: #00F5FF !important;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B0F2B 0%, #150A2E 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .sidebar-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: #00F5FF;
            margin-bottom: 4px;
        }

        .sidebar-section {
            font-size: 0.95rem;
            font-weight: 600;
            color: #B9C2FF;
            margin-top: 18px;
            margin-bottom: 6px;
        }

        .sidebar-item {
            font-size: 0.87rem;
            color: #9FA8DA;
            margin-bottom: 3px;
        }

        .sidebar-footer {
            margin-top: 40px;
            padding-top: 14px;
            border-top: 1px solid rgba(255,255,255,0.08);
            font-size: 0.82rem;
            color: #7C86B8;
            text-align: center;
        }

        /* ---------- Footer ---------- */
        .app-footer {
            text-align: center;
            padding: 30px 10px 15px 10px;
            margin-top: 40px;
            border-top: 1px solid rgba(255,255,255,0.08);
            color: #7C86B8;
            font-size: 0.85rem;
            line-height: 1.6;
        }

        /* ---------- Badges ---------- */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            margin: 3px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(0, 245, 255, 0.12);
            color: #00F5FF;
            border: 1px solid rgba(0, 245, 255, 0.3);
        }

        /* ---------- Animations ---------- */
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-15px);}
            to {opacity: 1; transform: translateY(0);}
        }

        @keyframes popIn {
            0% {opacity: 0; transform: scale(0.9);}
            100% {opacity: 1; transform: scale(1);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# HERO SECTION
# ==================================================================================

def hero_section():
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🎙️ EchoSense AI</div>
            <div class="hero-subtitle">Speech Emotion Recognition using Machine Learning</div>
            <div class="hero-description">
                Upload a speech recording — or record one live from your microphone — and
                let our AI model analyze vocal tone, pitch, and acoustic patterns to predict
                the speaker's underlying emotion, instantly and beautifully visualized.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# SIDEBAR
# ==================================================================================

def sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎙️ EchoSense AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item">Speech Emotion Recognition System</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">🛠️ Technology Stack</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="badge">Python</span>
            <span class="badge">Streamlit</span>
            <span class="badge">Librosa</span>
            <span class="badge">Scikit-Learn</span>
            <span class="badge">XGBoost</span>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section">🎤 Input Methods</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item">📤 Upload a .wav file</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-item">🎙️ Record live from microphone</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">🎭 Supported Emotions</div>', unsafe_allow_html=True)
        for emotion, emoji in EMOTION_EMOJI_MAP.items():
            st.markdown(f'<div class="sidebar-item">{emoji} {emotion}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">📖 About the Project</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-item">
            EchoSense AI is a machine learning system that identifies human
            emotions from speech audio. It extracts acoustic features such as
            MFCCs, pitch, and energy, then classifies emotions using trained
            ML models.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-footer">
                Developed by <strong>Varshitha</strong><br>
                B.Tech CSE • Machine Learning Project
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================================================
# INPUT SECTION (Upload OR Record)
# ==================================================================================

def upload_section():
    """
    Lets the user provide audio in one of two ways:
      1. Upload an existing .wav file
      2. Record audio live using their microphone (st.audio_input)

    Both paths return an audio object compatible with the rest of the pipeline
    (st.audio, .name, .size are all available on both UploadedFile-like objects).

    NOTE: st.audio_input requires Streamlit >= 1.36. If you're on an older
    version, either upgrade Streamlit or remove the "Record Audio" tab below.
    """
    st.markdown('<div class="section-header">📤 Provide Speech Recording</div>', unsafe_allow_html=True)

    tab_upload, tab_record = st.tabs(["📤 Upload File", "🎤 Record Audio"])

    audio_source = None

    with tab_upload:
        with st.container(border=True):
            st.markdown(
                '<p style="text-align:center; color:#9FA8DA; margin-bottom:10px;">'
                '🎵 Drag and drop a WAV file below, or click to browse</p>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Upload speech recording",
                type=["wav"],
                help="Only .wav audio files are supported.",
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                audio_source = uploaded_file

    with tab_record:
        with st.container(border=True):
            st.markdown(
                '<p style="text-align:center; color:#9FA8DA; margin-bottom:10px;">'
                '🎙️ Click the mic below, record your speech, then click it again to stop</p>',
                unsafe_allow_html=True,
            )
            recorded_audio = st.audio_input(
                "Record speech recording",
                help="Records audio directly from your microphone (browser permission required).",
                label_visibility="collapsed",
            )
            if recorded_audio is not None:
                # st.audio_input returns a BytesIO-like UploadedFile without a
                # user-chosen filename, so we give it a friendly, timestamped one.
                if not getattr(recorded_audio, "name", None):
                    recorded_audio.name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                audio_source = recorded_audio

    return audio_source


# ==================================================================================
# AUDIO PREVIEW SECTION
# ==================================================================================

def audio_preview_section(uploaded_file):
    st.markdown('<div class="section-header">🎧 Audio Preview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container(border=True):
            file_name = getattr(uploaded_file, "name", "Recorded Audio")
            st.markdown(f'<div class="card-title">📁 {file_name}</div>', unsafe_allow_html=True)
            st.audio(uploaded_file, format="audio/wav")

    with col2:
        with st.container(border=True):
            # ------------------------------------------------------------------
            # PLACEHOLDER: Duration can be computed using librosa in your backend.
            # Example:
            #   import librosa
            #   y, sr = librosa.load(uploaded_file)
            #   duration = librosa.get_duration(y=y, sr=sr)
            # ------------------------------------------------------------------
            duration_placeholder = "-- (compute via backend)"
            file_size_kb = round(uploaded_file.size / 1024, 2) if getattr(uploaded_file, "size", None) else "--"

            st.markdown(
                f"""
                <div class="card-title">ℹ️ File Info</div>
                <div class="card-text">🕒 Duration: {duration_placeholder}</div>
                <div class="card-text">💾 File Size: {file_size_kb} KB</div>
                <div class="card-text">📅 Captured: {datetime.now().strftime("%d %b %Y, %H:%M")}</div>
                """,
                unsafe_allow_html=True,
            )


# ==================================================================================
# PREDICTION SECTION
# ==================================================================================

def prediction_section(uploaded_file):
    st.markdown('<div class="section-header">🔮 Predict Emotion</div>', unsafe_allow_html=True)

    predict_clicked = st.button("✨ Predict Emotion ✨", use_container_width=True)

    result = None
    if predict_clicked:
        with st.spinner("Analyzing acoustic patterns... 🎶"):
            # --------------------------------------------------------------
            # PLACEHOLDER: Call your actual prediction pipeline here.
            # Example:
            #   result = predict_emotion(uploaded_file)
            # --------------------------------------------------------------
            time.sleep(1.5)  # Simulated processing delay for UX purposes only
            result = predict_emotion_placeholder(uploaded_file)

    return result


# ==================================================================================
# RESULTS SECTION
# ==================================================================================

def result_section(result):
    if result is None:
        return

    predicted_emotion, confidence = result
    emoji = EMOTION_EMOJI_MAP.get(predicted_emotion, "🎭")
    confidence_pct = int(confidence * 100)

    st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-emoji">{emoji}</div>
            <div class="result-emotion">{predicted_emotion}</div>
            <div class="result-confidence">Confidence Score: {confidence_pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(confidence)
    st.success(f"Prediction complete — the AI is {confidence_pct}% confident in '{predicted_emotion}'.")


# ==================================================================================
# VISUALIZATIONS SECTION
# ==================================================================================

def visualizations_section():
    st.markdown('<div class="section-header">📈 Audio Visualizations</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">🌊 Waveform</div>
                <div class="viz-placeholder">Insert waveform plot here<br>(e.g., librosa.display.waveshow)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # PLACEHOLDER: st.pyplot(waveform_figure)

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">🌈 Spectrogram</div>
                <div class="viz-placeholder">Insert spectrogram plot here<br>(e.g., librosa.display.specshow)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # PLACEHOLDER: st.pyplot(spectrogram_figure)

    with col3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">📶 Confidence Chart</div>
                <div class="viz-placeholder">Insert confidence bar chart here<br>(e.g., matplotlib bar chart of class probabilities)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # PLACEHOLDER: st.pyplot(confidence_figure)


# ==================================================================================
# INFORMATION CARDS SECTION
# ==================================================================================

def information_cards_section():
    st.markdown('<div class="section-header">💡 Learn More</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">⚙️ How It Works</div>
                <div class="card-text">
                1. Upload a WAV file, or record speech live from your microphone.<br>
                2. Acoustic features (MFCC, chroma, mel-spectrogram, etc.) are extracted.<br>
                3. A trained ML model (XGBoost / Scikit-Learn) classifies the emotion.<br>
                4. Results are displayed with confidence scores and visualizations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">📁 Supported Input</div>
                <div class="card-text">Upload a <strong>.wav</strong> file, or use the <strong>Record Audio</strong> tab to capture speech directly from your microphone for the most accurate feature extraction and prediction results.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">🧠 Model Information</div>
                <div class="card-text">
                The classification model is trained on labeled speech-emotion datasets
                using extracted MFCC and spectral features, with algorithms such as
                XGBoost and traditional Scikit-Learn classifiers for robust performance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">✅ Tips for Best Accuracy</div>
                <div class="card-text">
                • Use clear recordings with minimal background noise.<br>
                • Keep the speech duration between 2–10 seconds.<br>
                • Ensure only one speaker is present in the recording.<br>
                • Use a good quality microphone if possible.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================================================
# FOOTER
# ==================================================================================

def footer():
    st.markdown(
        f"""
        <div class="app-footer">
            <strong>🎙️ EchoSense AI</strong> — Speech Emotion Recognition<br>
            Machine Learning Project • IIITDM Kurnool<br>
            © {datetime.now().year} Varshitha. All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# MAIN APP FLOW
# ==================================================================================

def main():
    load_css()
    sidebar()
    hero_section()

    audio_source = upload_section()

    if audio_source is not None:
        audio_preview_section(audio_source)
        result = prediction_section(audio_source)
        result_section(result)
        visualizations_section()
    else:
        st.info("👆 Upload a WAV file or record audio above to begin emotion analysis.")

    information_cards_section()
    footer()


if __name__ == "__main__":
    main()