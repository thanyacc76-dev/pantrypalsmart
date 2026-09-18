import os
import time
import joblib
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Page Config
st.set_page_config(page_title="PantryPal", page_icon="🥑", layout="centered")

# Custom CSS for Breathing Glassmorphism, Animations, Floating Icons & Glowing Buttons
st.markdown(
    """
    <style>
    /* 1. Breathing Glassmorphic Background (Dark Blue, Green, Yellow, Purple, Maroon) */
    @keyframes breatheBackground {
        0%   { background: radial-gradient(circle at 20% 20%, #0d1b2a 0%, #081c15 50%, #000814 100%); }
        25%  { background: radial-gradient(circle at 80% 30%, #1a0c27 0%, #28051a 50%, #0a0112 100%); }
        50%  { background: radial-gradient(circle at 50% 80%, #1f1a00 0%, #0b251a 50%, #020617 100%); }
        75%  { background: radial-gradient(circle at 20% 70%, #2b0910 0%, #110d2c 50%, #03001e 100%); }
        100% { background: radial-gradient(circle at 20% 20%, #0d1b2a 0%, #081c15 50%, #000814 100%); }
    }

    .stApp {
        animation: breatheBackground 18s ease-in-out infinite alternate;
        color: #f8fafc;
        overflow-x: hidden;
    }

    /* 2. Floating Background Food Icons */
    .bg-icon {
        position: fixed;
        font-size: 2.2rem;
        opacity: 0.18;
        pointer-events: none;
        z-index: 0;
        animation: floatSlow 6s ease-in-out infinite alternate;
    }
    @keyframes floatSlow {
        0% { transform: translateY(0px) rotate(0deg); }
        100% { transform: translateY(-18px) rotate(12deg); }
    }

    /* 3. Title Hierarchy */
    .app-title-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .sub-brand {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.25em;
        color: #34d399;
        text-transform: uppercase;
        margin-bottom: 2px;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.5);
    }
    .main-brand {
        font-size: 3.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #a7f3d0 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
        filter: drop-shadow(0 0 15px rgba(255,255,255,0.2));
    }

    /* 4. Glowing Neon Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4), 0 0 35px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.7), 0 0 50px rgba(59, 130, 246, 0.5) !important;
    }

    /* 5. Glassmorphism Container Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 15px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.5);
        border: 2px dashed rgba(52, 211, 153, 0.4);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 20px;
    }

    .result-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .result-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #34d399;
        margin-top: 4px;
    }
    </style>

    <!-- Side Floating Food Icons -->
    <div class="bg-icon" style="top: 10%; left: 4%;">🥦</div>
    <div class="bg-icon" style="top: 28%; right: 5%;">🍎</div>
    <div class="bg-icon" style="top: 50%; left: 3%;">🥕</div>
    <div class="bg-icon" style="top: 72%; right: 4%;">🥑</div>
    <div class="bg-icon" style="top: 88%; left: 6%;">🍇</div>
""",
    unsafe_allow_html=True,
)


# Model Loading
@st.cache_resource
def load_assets():
    assets = {}
    try:
        assets["food_model"] = tf.keras.models.load_model("food_model.keras")
        assets["freshness_model"] = tf.keras.models.load_model(
            "freshness_model.keras"
        )
        assets["expiry_model"] = joblib.load("expiry_model.joblib")

        with open("class_names.txt", "r") as f:
            assets["food_classes"] = [line.strip() for line in f.readlines()]

        with open("freshness_class_names.txt", "r") as f:
            assets["freshness_classes"] = [
                line.strip() for line in f.readlines()
            ]

        if os.path.exists("food_encoder.joblib"):
            assets["food_encoder"] = joblib.load("food_encoder.joblib")
        if os.path.exists("status_encoder.joblib"):
            assets["status_encoder"] = joblib.load("status_encoder.joblib")
        if os.path.exists("storage_encoder.joblib"):
            assets["storage_encoder"] = joblib.load("storage_encoder.joblib")

    except Exception as e:
        st.error(f"Error loading models: {e}")
    return assets


assets = load_assets()

# Initialize Session State Navigation
if "page" not in st.session_state:
    st.session_state.page = "landing"

# ================= PAGE 1: LANDING PAGE =================
if st.session_state.page == "landing":
    st.markdown(
        """
        <div class="app-title-container">
            <div class="sub-brand">SMART STORAGE AI</div>
            <div class="main-brand">PantryPal</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 30px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-bottom: 8px;">
                Hey, what are we cooking today? 🍳
            </h2>
            <p style="color: #94a3b8; font-size: 1rem; margin-0;">
                Let's inspect your ingredients and see what's fresh!
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Let's Go! 🚀", use_container_width=True):
            # Falling Food Animation Trigger
            st.snow()  # Streamlit animation effect
            st.toast("🍇 🍎 🥦 Gathering your ingredients... 🥑 🥕 🌽")
            time.sleep(1.8)
            st.session_state.page = "app"
            st.rerun()

# ================= PAGE 2: MAIN INSPECTION APP =================
elif st.session_state.page == "app":
    st.markdown(
        """
        <div class="app-title-container">
            <div class="sub-brand">SMART STORAGE AI</div>
            <div class="main-brand">PantryPal</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Back to Landing option
    if st.button("← Back to Landing", use_container_width=False):
        st.session_state.page = "landing"
        st.rerun()

    st.markdown("### Upload or Capture an Ingredient")
    uploaded_file = st.file_uploader(
        "Choose a food photo...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )

        if st.button("Inspect Item 🔍", use_container_width=True):
            with st.spinner("Analyzing item with AI models..."):
                try:
                    # Preprocess Image
                    image = Image.open(uploaded_file).convert("RGB")
                    image_resized = ImageOps.fit(
                        image, (224, 224), Image.Resampling.LANCZOS
                    )
                    img_array = np.asarray(image_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Model Predictions
                    food_preds = assets["food_model"].predict(img_array)
                    detected_food = assets["food_classes"][
                        np.argmax(food_preds[0])
                    ]

                    freshness_preds = assets["freshness_model"].predict(
                        img_array
                    )
                    freshness_status = assets["freshness_classes"][
                        np.argmax(freshness_preds[0])
                    ]

                    # 5-Feature Expiry Prediction
                    if "food_encoder" in assets and "status_encoder" in assets:
                        food_enc = assets["food_encoder"].transform(
                            [detected_food]
                        )[0]
                        status_enc = assets["status_encoder"].transform(
                            [freshness_status]
                        )[0]
                        storage_enc = (
                            assets["storage_encoder"].transform(
                                ["refrigerator"]
                            )[0]
                            if "storage_encoder" in assets
                            else 0
                        )

                        features = np.array(
                            [[food_enc, status_enc, storage_enc, 4.0, 70.0]]
                        )
                        estimated_expiry = int(
                            assets["expiry_model"].predict(features)[0]
                        )
                    else:
                        estimated_expiry = 5

                    # Render Results
                    st.markdown("### Analysis Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"""
                        <div class="glass-card">
                            <div class="result-label">Food Detected</div>
                            <div class="result-val" style="color: #ffffff;">{detected_food.title()}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.markdown(
                            f"""
                        <div class="glass-card">
                            <div class="result-label">Freshness</div>
                            <div class="result-val">{freshness_status.title()}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f"""
                    <div class="glass-card">
                        <div class="result-label">Est. Remaining Shelf Life</div>
                        <div class="result-val" style="color: #6EE7B7;">{max(0, estimated_expiry)} Days</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"Inference error: {e}")
