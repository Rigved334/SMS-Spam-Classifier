import os
import numpy as np
import joblib
import streamlit as st

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered",
)

# ──────────────────────────────────────────────
# Load the pre-trained pipeline (cached so it
# is only read from disk once per session)
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "SMS_Spam_Predictor.pkl",
)

THRESHOLD = 0.19  # same threshold used in the original classifier


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


pipeline = load_model()

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── typography ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── header area ───────────────────────── */
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .header-container h1 {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .header-container p {
        opacity: 0.65;
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* ── result cards ──────────────────────── */
    .result-card {
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.4rem;
        text-align: center;
        transition: transform 0.15s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }
    .spam-card {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1px solid #f87171;
    }
    .ham-card {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #4ade80;
    }
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-prob {
        font-size: 0.9rem;
        color: #374151;
    }

    /* ── message preview ───────────────────── */
    .msg-preview {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.88rem;
        line-height: 1.55;
        word-wrap: break-word;
    }

    /* ── info section ──────────────────────── */
    .info-section {
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
        font-size: 0.82rem;
        opacity: 0.55;
        text-align: center;
    }

    /* ── hide default Streamlit branding ────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="header-container">
        <h1>📩 SMS Spam Classifier</h1>
        <p>Paste a text message below and hit <strong>Classify</strong> to check if it's spam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Input area
# ──────────────────────────────────────────────
message = st.text_area(
    "Enter SMS message",
    height=140,
    placeholder="e.g.  Congratulations! You've won a £1000 gift card. Call now to claim...",
    label_visibility="collapsed",
)

classify_clicked = st.button("Classify", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Prediction logic (mirrors original script)
# ──────────────────────────────────────────────
if classify_clicked:
    if not message.strip():
        st.warning("Please enter a message to classify.")
    else:
        spam_probability = pipeline.predict_proba([message])[:, 1][0]
        prediction = "spam" if spam_probability >= THRESHOLD else "ham"

        if prediction == "spam":
            card_class = "spam-card"
            emoji = "🚫"
            label_text = "Spam"
            label_color = "#dc2626"
        else:
            card_class = "ham-card"
            emoji = "✅"
            label_text = "Not Spam"
            label_color = "#16a34a"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label" style="color:{label_color};">
                    {emoji} {label_text}
                </div>
                <div class="result-prob">
                    Spam probability: <strong>{spam_probability:.1%}</strong>
                </div>
            </div>
            <div class="msg-preview">
                <strong>Message:</strong> {message}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────
# Footer info
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="info-section">
        Model: TF-IDF (char 3-5 grams) + Logistic Regression
    </div>
    """,
    unsafe_allow_html=True,
)
