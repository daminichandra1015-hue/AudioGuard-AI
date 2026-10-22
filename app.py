
from pathlib import Path
import tempfile

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

# =========================
# 1. PROJECT SETTINGS
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_model.pkl"

st.set_page_config(
    page_title="AudioGuard AI",
    page_icon="🎙️",
    layout="wide"
)


# =========================
# 2. CUSTOM DESIGN
# =========================

st.markdown("""
<style>
    .stApp {
        background-color: #f6f5ff;
    }

    h1, h2, h3 {
        color: #44318d;
    }

    .main-title {
        text-align: center;
        color: #5b45c6;
        font-size: 42px;
        font-weight: 800;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 17px;
        margin-bottom: 28px;
    }

    .info-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e6e0ff;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e6e0ff;
    }

    section[data-testid="stFileUploader"] {
        background: white;
        padding: 16px;
        border-radius: 14px;
        border: 1px dashed #8e7bea;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# 3. HEADER
# =========================

st.markdown(
    '<div class="main-title">🎙️ AudioGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Audio Deepfake Detection Using MFCC'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload an audio file to explore its acoustic features "
    "and analyse its MFCC representation."
)

st.divider()


# =========================
# 4. AUDIO FEATURE EXTRACTION
# =========================

def extract_features(audio_path):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The uploaded audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    # Mean MFCC values: 13 features
    features = np.mean(mfcc, axis=1)

    duration = len(audio) / sample_rate

    return audio, features, mfcc, duration, sample_rate


# =========================
# 5. AUDIO UPLOAD
# =========================

st.subheader("📂 Upload Audio")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac", "ogg"]
)

if uploaded_file is None:
    st.info("Upload an audio file to begin the analysis.")

    st.markdown("""
    ### How it works
    1. Upload an audio recording.
    2. Extract acoustic features using MFCC.
    3. View the waveform and MFCC heatmap.
    4. Get a Real/Fake prediction only when a trained model
       is available.
    """)

    st.stop()


# =========================
# 6. PROCESS AUDIO
# =========================

st.subheader("🔊 Uploaded Audio")

st.audio(uploaded_file.getvalue())

st.write("**File name:**", uploaded_file.name)
st.write(
    "**File size:**",
    f"{uploaded_file.size / (1024 * 1024):.2f} MB"
)

audio = None
features = None
mfcc = None
duration = None
sample_rate = None

try:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)

    try:
        (
            audio,
            features,
            mfcc,
            duration,
            sample_rate
        ) = extract_features(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

except Exception as error:
    st.error(
        "Unable to analyse this audio file. "
        "Please try a valid WAV, MP3, FLAC or OGG file."
    )
    st.caption(f"Technical details: {error}")
    st.stop()


# =========================
# 7. AUDIO INFORMATION
# =========================

st.divider()
st.subheader("📊 Audio Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Duration",
        f"{duration:.2f} sec"
    )

with col2:
    st.metric(
        "Sample Rate",
        f"{sample_rate:,} Hz"
    )

with col3:
    st.metric(
        "MFCC Features",
        len(features)
    )

st.caption(
    "The uploaded audio is converted to mono and resampled "
    "to 16,000 Hz for feature extraction."
)


# =========================
# 8. WAVEFORM
# =========================

st.divider()
st.subheader("〰️ Audio Waveform")

fig_wave, ax_wave = plt.subplots(figsize=(12, 3))

time_axis = np.arange(len(audio)) / sample_rate

ax_wave.plot(time_axis, audio, linewidth=0.6)
ax_wave.set_xlabel("Time (seconds)")
ax_wave.set_ylabel("Amplitude")
ax_wave.set_title("Audio Waveform")
ax_wave.grid(alpha=0.2)

fig_wave.tight_layout()
st.pyplot(fig_wave)
plt.close(fig_wave)


# =========================
# 9. MFCC HEATMAP
# =========================

st.divider()
st.subheader("🎨 MFCC Feature Visualisation")

fig_mfcc, ax_mfcc = plt.subplots(figsize=(12, 4))

image = ax_mfcc.imshow(
    mfcc,
    aspect="auto",
    origin="lower",
    interpolation="nearest"
)

ax_mfcc.set_xlabel("Time Frames")
ax_mfcc.set_ylabel("MFCC Coefficient")
ax_mfcc.set_title("Mel-Frequency Cepstral Coefficients")

fig_mfcc.colorbar(
    image,
    ax=ax_mfcc,
    label="Coefficient Value"
)

fig_mfcc.tight_layout()
st.pyplot(fig_mfcc)
plt.close(fig_mfcc)


# =========================
# 10. MFCC VALUES
# =========================

with st.expander("View extracted MFCC values"):
    for index, value in enumerate(features):
        st.write(
            f"MFCC {index + 1}: {value:.4f}"
        )


# =========================
# 11. REAL / FAKE PREDICTION
# =========================

st.divider()
st.subheader("🛡️ Deepfake Detection Result")

if MODEL_PATH.exists():

    try:
        model = joblib.load(MODEL_PATH)

        # This model must be trained using the same
        # 13 mean-MFCC features.
        prediction = int(
            model.predict(
                features.reshape(1, -1)
            )[0]
        )

        if prediction == 0:
            st.success("Model prediction: REAL AUDIO")
        elif prediction == 1:
            st.error("Model prediction: FAKE AUDIO")
        else:
            st.warning(
                f"Unexpected model label: {prediction}"
            )

        st.caption(
            "This is a model prediction, not a guarantee. "
            "Its reliability depends on the quality and testing "
            "of the labelled training data."
        )

    except Exception as error:
        st.error(
            "The model could not make a prediction. "
            "Check that it was trained with the same 13 "
            "mean-MFCC features."
        )
        st.caption(f"Technical details: {error}")

else:
    st.warning("Real/Fake prediction is not ready yet.")

    st.write(
        "The audio analysis is working, but the detection model "
        "has not been created yet. Verified real and fake audio "
        "labels are needed before training and evaluating it."
    )


# =========================
# 12. FOOTER
# =========================

st.divider()

st.caption(
    "AudioGuard AI | Educational Mini Project | "
    "MFCC-Based Audio Analysis"
)
