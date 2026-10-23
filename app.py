
from pathlib import Path
import tempfile

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_model.pkl"

ICE_SILVER = "#C9D0D8"
DEEP_AUBERGINE = "#351A35"
MARIGOLD = "#E8A52A"
DARK_BG = "#17131C"
CARD_BG = "#211B27"
MUTED_TEXT = "#AFA7B8"

st.set_page_config(
    page_title="AudioGuard AI | Audio Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. PREMIUM DESIGN
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --silver: #C9D0D8;
    --aubergine: #351A35;
    --marigold: #E8A52A;
    --background: #17131C;
    --card: #211B27;
    --muted: #AFA7B8;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, #38213B 0%, transparent 38%),
        radial-gradient(ellipse at 95% 20%, #292037 0%, transparent 34%),
        var(--background);
    color: var(--silver);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.3rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--silver) !important;
    font-family: 'Manrope', sans-serif !important;
}

p, label, li, .stCaption {
    color: var(--silver);
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 25px;
}

.brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 54px;
    height: 54px;
    border-radius: 17px;
    background: linear-gradient(135deg, #563657, #281D30);
    border: 1px solid #765477;
    font-size: 27px;
}

.brand-name {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: var(--silver);
    letter-spacing: -0.7px;
}

.brand-tag {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 38px 34px;
    border-radius: 25px;
    background: linear-gradient(125deg, #39243C, #251C2D 60%, #292039);
    border: 1px solid #5C435F;
    margin-bottom: 26px;
}

.hero:after {
    content: "♫";
    position: absolute;
    right: 38px;
    top: -28px;
    font-size: 180px;
    color: rgba(232, 165, 42, 0.09);
    pointer-events: none;
}

.eyebrow {
    color: var(--marigold);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    color: #F1EDF3;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(30px, 4.2vw, 48px);
    line-height: 1.16;
    font-weight: 800;
    letter-spacing: -1.6px;
    margin-bottom: 14px;
}

.hero-copy {
    max-width: 650px;
    color: #D0C5D4;
    font-size: 16px;
    line-height: 1.8;
}

.quote {
    color: var(--marigold);
    font-size: 15px;
    font-weight: 600;
    font-style: italic;
    margin-top: 23px;
}

.section-kicker {
    color: var(--marigold);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.section-heading {
    font-size: 23px;
    font-weight: 800;
    color: var(--silver);
    margin-bottom: 5px;
}

.section-copy {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 18px;
}

.upload-panel {
    background: linear-gradient(145deg, #29202F, #201A26);
    border: 1px solid #55405A;
    border-radius: 22px;
    padding: 22px;
    margin-bottom: 22px;
}

.upload-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.upload-title {
    color: #F1EDF3;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
}

.upload-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

div[data-testid="stFileUploader"] {
    background: #1A1520;
    border: 1px dashed #806486;
    border-radius: 15px;
    padding: 12px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
}

div[data-testid="stFileUploader"] button {
    background: var(--marigold) !important;
    color: #241726 !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

div[data-testid="stFileUploader"] small {
    color: var(--muted) !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #2A2230, #211B27);
    border: 1px solid #4B394F;
    padding: 20px;
    border-radius: 17px;
    min-height: 120px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--silver) !important;
    font-weight: 800;
}

div[data-testid="stAudio"] {
    border-radius: 14px;
}

div[data-testid="stAlert"] {
    border-radius: 13px;
}

.stSuccess {
    background: #17352B !important;
}

.stError {
    background: #40212B !important;
}

.stInfo {
    background: #282037 !important;
}

div[data-testid="stExpander"] {
    background: #211B27;
    border: 1px solid #4B394F;
    border-radius: 13px;
}

hr {
    border-color: #443549 !important;
}

div[data-testid="stVerticalBlock"] > div:has(> .result-card) {
    width: 100%;
}

.result-card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(145deg, #29202F, #211B27);
    border: 1px solid #55405A;
    margin: 10px 0 18px 0;
}

.result-label {
    color: var(--marigold);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-title {
    color: #F1EDF3;
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.result-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
    margin-top: 8px;
}

.feature-card {
    background: #211B27;
    border: 1px solid #443549;
    border-radius: 17px;
    padding: 19px;
    height: 100%;
}

.feature-number {
    color: var(--marigold);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.feature-title {
    color: var(--silver);
    font-size: 16px;
    font-weight: 700;
    margin: 9px 0 7px 0;
}

.feature-copy {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.65;
}

.footer {
    text-align: center;
    color: #92879A;
    font-size: 12px;
    line-height: 1.8;
    padding: 20px 0 5px 0;
}

.stButton button {
    background: var(--marigold);
    color: #241726;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 27px 22px;
    }

    .hero:after {
        right: 8px;
        font-size: 115px;
    }

    .hero-title {
        font-size: 31px;
    }

    .brand-name {
        font-size: 22px;
    }
}

/* ===== READABILITY & CONTRAST FIX ===== */

/* Default text and labels */
.stApp,
.stApp p,
.stApp li,
.stApp label,
.stApp span,
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown label {
    color: #E3DDE8;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #E5E9EF !important;
}

/* Sidebar and form labels */
[data-testid="stSidebar"],
[data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] label,
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label {
    color: #E3DDE8 !important;
}

/* Upload area text */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] p {
    color: #E3DDE8 !important;
}

/* File uploader buttons */
[data-testid="stFileUploader"] button {
    background: #E8A52A !important;
    color: #241726 !important;
    border: none !important;
    font-weight: 700 !important;
}

/* Secondary and helper text */
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
small,
.stMarkdown small {
    color: #BDB3C7 !important;
}

/* Metrics */
[data-testid="stMetricLabel"] {
    color: #C9D0D8 !important;
}

[data-testid="stMetricValue"] {
    color: #F0EAF3 !important;
}

/* Audio player container */
[data-testid="stAudio"] {
    background: #29202F;
    border: 1px solid #55405A;
    border-radius: 14px;
    padding: 6px;
}

/* Expanders */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #E5E9EF !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
[data-baseweb="select"] > div {
    background-color: #211B27 !important;
    color: #F0EAF3 !important;
    border-color: #806486 !important;
}

/* Keyboard focus */
.stApp button:focus-visible,
.stApp input:focus-visible {
    outline: 2px solid #E8A52A !important;
    outline-offset: 2px;
}

/* Links */
.stApp a {
    color: #F2BD58 !important;
}

/* Alerts */
[data-testid="stAlert"] p {
    color: #F0EAF3 !important;
}

/* ===== BIG & CLEAR FINAL RESULT ===== */

.result-card {
    padding: 32px !important;
    border-radius: 22px !important;
    margin: 18px 0 22px !important;
    border: 2px solid #806486 !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
}

.result-label {
    font-size: 14px !important;
    letter-spacing: 2.5px !important;
    font-weight: 800 !important;
}

.result-title {
    font-size: clamp(36px, 5vw, 58px) !important;
    line-height: 1.2 !important;
    font-weight: 900 !important;
    margin-top: 14px !important;
    letter-spacing: -1px !important;
}

.result-copy {
    font-size: 16px !important;
    line-height: 1.8 !important;
    margin-top: 14px !important;
    color: #E3DDE8 !important;
}

/* Make the final Streamlit status message readable too */
[data-testid="stAlert"] {
    padding: 16px 20px !important;
    border-radius: 14px !important;
}

[data-testid="stAlert"] p {
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #F5F0F7 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. REUSABLE HELPERS
# ==========================================

def section_heading(kicker, title, description=""):
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-heading">{title}</div>
        <div class="section-copy">{description}</div>
        """,
        unsafe_allow_html=True
    )


def style_plot(ax, fig):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    ax.tick_params(colors=ICE_SILVER, labelsize=9)
    ax.xaxis.label.set_color(ICE_SILVER)
    ax.yaxis.label.set_color(ICE_SILVER)
    ax.title.set_color(ICE_SILVER)

    for spine in ax.spines.values():
        spine.set_color("#55405A")

    ax.grid(color="#55405A", alpha=0.35)


# ==========================================
# 4. HEADER AND HERO
# ==========================================

st.markdown("""
<div class="brand">
    <div class="brand-icon">🎙️</div>
    <div>
        <div class="brand-name">AudioGuard AI</div>
        <div class="brand-tag">Audio Intelligence Lab</div>
    </div>
</div>

<div class="hero">
    <div class="eyebrow">AI-powered audio analysis</div>
    <div class="hero-title">
        Every voice has a story.<br>
        Discover what lies beneath.
    </div>
    <div class="hero-copy">
        Explore acoustic patterns, visualize MFCC features, and
        examine a model-based real-or-fake audio prediction in
        one intelligent workspace.
    </div>
    <div class="quote">
        “Listen beyond the sound. Discover the truth.”
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. FEATURE EXTRACTION
# ==========================================

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

    features = np.mean(mfcc, axis=1)
    duration = len(audio) / sample_rate

    return audio, features, mfcc, duration, sample_rate


# ==========================================
# 6. AUDIO UPLOAD
# ==========================================

section_heading(
    "STEP 01 · INPUT",
    "Bring your audio into focus",
    "Start with a recording. We'll take care of the acoustic analysis."
)

st.markdown("""
<div class="upload-panel">
    <div class="upload-icon">✦</div>
    <div class="upload-title">Drop your audio into the spotlight</div>
    <div class="upload-copy">
        Choose an audio recording to inspect its waveform, acoustic
        fingerprint and model prediction. Your analysis starts here.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    help="Supported extensions: WAV, MP3,MPA, FLAC and OGG."
)

if uploaded_file is None:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-number">HOW IT WORKS</div>
        <div class="feature-title">From sound to insight</div>
        <div class="feature-copy">
            01 — Upload an audio recording.<br>
            02 — Extract 13 mean MFCC features.<br>
            03 — Explore waveform and MFCC visualizations.<br>
            04 — Review the trained model's prediction, if available.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        AUDIOGUARD AI · EDUCATIONAL PROJECT<br>
        Every prediction should be interpreted with care.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ==========================================
# 7. PROCESS AUDIO
# ==========================================

section_heading(
    "STEP 02 · LISTEN",
    "Your audio workspace",
    "Review the uploaded recording and its file details."
)

st.markdown(
    f"""
    <div class="feature-card">
        <div class="feature-number">SELECTED RECORDING</div>
        <div class="feature-title">🎧 {uploaded_file.name}</div>
        <div class="feature-copy">
            File size: {uploaded_file.size / (1024 * 1024):.2f} MB
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.audio(uploaded_file.getvalue())

audio = None
features = None
mfcc = None
duration = None
sample_rate = None

temp_path = None

try:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)

    (
        audio,
        features,
        mfcc,
        duration,
        sample_rate
    ) = extract_features(str(temp_path))

except Exception as error:
    st.error(
        "Unable to analyse this audio. Please try a valid "
        "WAV, MP3, FLAC or OGG recording."
    )
    st.caption(f"Technical details: {error}")

    st.stop()

finally:
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)


# ==========================================
# 8. AUDIO METRICS
# ==========================================

st.divider()

section_heading(
    "STEP 03 · INSPECT",
    "Acoustic overview",
    "A quick snapshot of the recording being analysed."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("DURATION", f"{duration:.2f} sec")

with col2:
    st.metric("SAMPLE RATE", f"{sample_rate:,} Hz")

with col3:
    st.metric("MFCC FEATURES", len(features))

st.caption(
    "Audio is converted to mono and resampled to 16,000 Hz "
    "before MFCC feature extraction."
)


# ==========================================
# 9. WAVEFORM
# ==========================================

st.divider()

section_heading(
    "VISUALIZATION 01",
    "The waveform",
    "Amplitude over time reveals the recording's acoustic shape."
)

fig_wave, ax_wave = plt.subplots(figsize=(12, 3.5))

time_axis = np.arange(len(audio)) / sample_rate

ax_wave.plot(
    time_axis,
    audio,
    color=MARIGOLD,
    linewidth=0.65
)

ax_wave.set_xlabel("Time (seconds)")
ax_wave.set_ylabel("Amplitude")
ax_wave.set_title("Audio Signal · Amplitude vs Time")

style_plot(ax_wave, fig_wave)
fig_wave.tight_layout()

st.pyplot(fig_wave, use_container_width=True)
plt.close(fig_wave)


# ==========================================
# 10. MFCC HEATMAP
# ==========================================

st.divider()

section_heading(
    "VISUALIZATION 02",
    "The acoustic fingerprint",
    "MFCC coefficients summarize important spectral characteristics "
    "across the recording."
)

fig_mfcc, ax_mfcc = plt.subplots(figsize=(12, 4.2))

image = ax_mfcc.imshow(
    mfcc,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    cmap="magma"
)

ax_mfcc.set_xlabel("Time Frames")
ax_mfcc.set_ylabel("MFCC Coefficient")
ax_mfcc.set_title("Mel-Frequency Cepstral Coefficients")

style_plot(ax_mfcc, fig_mfcc)

colorbar = fig_mfcc.colorbar(
    image,
    ax=ax_mfcc,
    pad=0.02
)
colorbar.ax.yaxis.set_tick_params(color=ICE_SILVER)
plt.setp(colorbar.ax.get_yticklabels(), color=ICE_SILVER)
colorbar.set_label("Coefficient Value", color=ICE_SILVER)

fig_mfcc.tight_layout()

st.pyplot(fig_mfcc, use_container_width=True)
plt.close(fig_mfcc)


# ==========================================
# 11. MFCC VALUES
# ==========================================

with st.expander("⌁  Explore the 13 extracted MFCC values"):
    for index, value in enumerate(features):
        st.write(f"**MFCC {index + 1}:** {value:.4f}")


# ==========================================
# 12. REAL / FAKE PREDICTION
# ==========================================

st.divider()

section_heading(
    "STEP 04 · RESULT",
    "Audio authenticity assessment",
    "Review the prediction returned by your trained model."
)

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)

        prediction = int(
            model.predict(features.reshape(1, -1))[0]
        )

        if prediction == 0:
            st.markdown("""
            <div class="result-card">
                <div class="result-label">MODEL OUTPUT · LABEL 0</div>
                <div class="result-title">✓ REAL AUDIO</div>
                <div class="result-copy">
                    The trained model classified this recording as real.
                    This result is a prediction, not proof of authenticity.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.success("Model prediction: REAL AUDIO")

        elif prediction == 1:
            st.markdown("""
            <div class="result-card">
                <div class="result-label">MODEL OUTPUT · LABEL 1</div>
                <div class="result-title">⚠ Potentially Fake Audio</div>
                <div class="result-copy">
                    The trained model classified this recording as fake.
                    Independent verification is needed before drawing conclusions.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.error("Model prediction: FAKE AUDIO")

        else:
            st.warning(f"Unexpected model label: {prediction}")

        st.caption(
            "Model predictions can be wrong. Reliability depends on "
            "the training data, labels, evaluation and audio conditions. "
            "No confidence score is shown because this app does not "
            "currently calculate one."
        )

    except Exception as error:
        st.error(
            "The model could not make a prediction. Check that it "
            "was trained with the same 13 mean-MFCC features."
        )
        st.caption(f"Technical details: {error}")

else:
    st.warning("The trained model file was not found.")

    st.write(
        "Audio analysis and visualizations are available, but a "
        "prediction requires deepfake_model.pkl in the project folder."
    )


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">
    <b style="color:#C9D0D8;">AUDIOGUARD AI</b><br>
    Listen beyond the sound. Discover the truth.<br>
    Educational Mini Project · MFCC-Based Audio Analysis
</div>
""", unsafe_allow_html=True)
