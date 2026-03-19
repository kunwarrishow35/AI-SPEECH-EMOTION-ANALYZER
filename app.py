import io
import time
import streamlit as st

from model import EmotionDetectionModel, EMOTION_LABELS
from utils import (
    load_audio,
    plot_probability_distribution,
    get_emotion_suggestion,
    apply_custom_css
)

st.set_page_config(
    page_title="AI Speech Emotion Analyzer",
    page_icon="🎙️",
    layout="centered"
)

apply_custom_css()


@st.cache_resource
def load_system_model():
    return EmotionDetectionModel()


st.title("🎙️ AI Speech Emotion Analyzer")

st.markdown("""
<p style='color: #5b6a7a; font-size: 1.1rem; margin-bottom: 2rem;'>
Upload an audio file or record speech to analyze the underlying emotion.
</p>
""", unsafe_allow_html=True)


with st.container():
    try:
        with st.spinner("🤖 Loading model..."):
            model = load_system_model()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()

    st.markdown("### 📥 Input Source")
    tabs = st.tabs(["📁 Upload File", "🎤 Record Audio"])

    audio_bytes = None

    with tabs[0]:
        uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "ogg"],
        help="Supported formats: WAV, MP3, OGG"
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()

    with tabs[1]:
        st.info("Record a short voice note (allow microphone access).")
        audio_value = st.audio_input("Record from browser:")
        if audio_value is not None:
            audio_bytes = audio_value.read()


if audio_bytes is not None:
    st.markdown("---")
    st.audio(audio_bytes, format='audio/wav')

    if st.button("✨ Predict Emotion", use_container_width=True):

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            remaining_time = round((100 - i) * 0.035, 1)
            status_text.markdown(
                f"<p style='color: #00d4ff; font-weight: 600;'>"
                f"Analyzing audio... ~{remaining_time}s remaining"
                f"</p>",
                unsafe_allow_html=True
            )
            progress_bar.progress(i + 1)
            time.sleep(0.035)

        status_text.empty()
        progress_bar.empty()

        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_array, sr = load_audio(audio_file)

            emotion, confidence, probabilities = model.predict(audio_array, sr)

            st.markdown("### 📊 Analysis Overview")
            col_res1, col_res2 = st.columns([1, 1.2])

            with col_res1:
                st.metric(
                    label="Dominant Emotion",
                    value=emotion,
                    delta=f"{confidence*100:.2f}% Confidence",
                    delta_color="normal"
                )

                st.markdown("#### 💡 Note")
                st.success(get_emotion_suggestion(emotion))

            with col_res2:
                st.markdown(
                    "<p style='text-align:center; font-weight: 600;'>Probability Distribution</p>",
                    unsafe_allow_html=True
                )

                fig = plot_probability_distribution(probabilities, EMOTION_LABELS)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'displayModeBar': False}
                )

        except Exception:
            import traceback
            st.error("⚠️ Failed to process audio. Try another sample.")
            st.code(traceback.format_exc())


st.markdown(
    "<div class='footer'>Made by <span>Team Visionaries</span></div>",
    unsafe_allow_html=True
)