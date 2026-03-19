import librosa
import pandas as pd
import plotly.express as px
import streamlit as st


import tempfile
import os

def load_audio(file_path_or_bytes, target_sr=16000):
    """Load audio and resample to the target sample rate."""
    if hasattr(file_path_or_bytes, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(file_path_or_bytes.read())
            tmp_path = tmp_file.name
        try:
            audio, sr = librosa.load(tmp_path, sr=target_sr)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        audio, sr = librosa.load(file_path_or_bytes, sr=target_sr)
    return audio, sr


def plot_probability_distribution(probabilities, labels):
    """Create a horizontal bar chart for emotion probabilities."""
    df = pd.DataFrame({
        "Emotion": labels,
        "Probability (%)": [p * 100 for p in probabilities]
    }).sort_values(by="Probability (%)", ascending=True)

    fig = px.bar(
        df,
        x="Probability (%)",
        y="Emotion",
        orientation="h",
        color="Probability (%)",
        color_continuous_scale="Blues_r"
    )

    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title="",
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=300,
        font={"size": 14, "color": "#e2e8f0"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def get_emotion_suggestion(emotion):
    """Return a short suggestion based on detected emotion."""
    suggestions = {
        "Happy": "Keep spreading positivity! 🌟 Share your good energy with someone today.",
        "Sad": "It's okay to feel down sometimes. Try calming music or a short walk. 🎧",
        "Angry": "Take a deep breath. Step away for a moment and reset. 🌿",
        "Neutral": "You seem balanced. A good time to stay focused and productive. ⚖️",
        "Fearful": "You're safe. Try grounding exercises or talk to someone you trust. 🧘",
        "Surprised": "Take a moment to process what's happening around you. 😲"
    }

    return suggestions.get(emotion, "Stay mindful and take care of yourself. 💙")


def apply_custom_css():
    """Apply custom dark theme styling to the Streamlit app."""
    st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
    }

    div.css-1r6slb0, div.css-12oz5g7 {
        background-color: #1a1e26;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 30px rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.1);
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00d4ff 0%, #007bff 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #007bff 0%, #00d4ff 100%);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
        transform: translateY(-2px);
    }

    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }

    p, stMarkdown {
        color: #a0aec0 !important;
    }

    .stAlert {
        border-radius: 10px;
        background-color: #1a1e26 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .footer {
        text-align: center;
        padding-top: 50px;
        padding-bottom: 20px;
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 1px;
    }

    .footer span {
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 0 5px rgba(0, 212, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)