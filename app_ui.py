import os
import time
import joblib
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Page Config
st.set_page_config(page_title="PantryPal", page_icon="🥑", layout="centered")

# Custom CSS for Baby Pink Base, Pastel Breathing Gradient, Falling Food, and Maroon-Pink Buttons
st.markdown(
    """
    <style>
    /* 1. Light Baby Pink Background + Subtle Pastel Breathing Glow */
    @keyframes pastelBreathe {
        0%   { background: radial-gradient(circle at 20% 20%, #ffe4e6 0%, #fff0f3 60%, #fff5f7 100%); }
        25%  { background: radial-gradient(circle at 80% 30%, #f3e8ff 0%, #fae8ff 60%, #fff5f7 100%); }
        50%  { background: radial-gradient(circle at 50% 80%, #dcfce7 0%, #f0fdf4 60%, #fff5f7 100%); }
        75%  { background: radial-gradient(circle at 20% 70%, #fce7f3 0%, #fee2e2 60%, #fff5f7 100%); }
        100% { background: radial-gradient(circle at 20% 20%, #ffe4e6 0%, #fff0f3 60%, #fff5f7 100%); }
    }

    .stApp {
        animation: pastelBreathe 20s ease-in-out infinite alternate;
        color: #4a041f;
        background-color: #fff5f7;
        overflow-x: hidden;
    }

    /* 2. Side Floating Food Icons */
    .bg-icon {
        position: fixed;
        font-size: 2.2rem;
        opacity: 0.35;
        pointer-events: none;
        z-index: 0;
        animation: floatSlow 6s ease-in-out infinite alternate;
    }
    @keyframes floatSlow {
        0% { transform: translateY(0px) rotate(0deg); }
        100% { transform: translateY(-16px) rotate(10deg); }
    }

    /* 3. Title Hierarchy & Typography */
    .app-title-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .sub-brand {
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.25em;
        color: #9f1239;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .main-brand {
        font-size: 3.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #881337 0%, #be123c 50%, #fb7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
        filter: drop-shadow(0 4px 10px rgba(190, 18, 60, 0.15));
    }

    /* 4. Maroon to Light Pink Glowing Gradient Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #881337 0%, #be123c 50%, #f472b6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 28px !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(136, 19, 55, 0.3), 0 0 15px rgba(244, 114, 182, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 12px 28px rgba(136, 19, 55, 0.4), 0 0 25px rgba(244, 114, 182, 0.6) !important;
    }

    /* 5. Glassmorphism Cards (Pastel Themed) */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(251, 113, 133, 0.25);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(136, 19, 55, 0.08);
        margin-top: 15px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.65);
        border: 2px dashed rgba(225, 29, 72, 0.35);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 20px;
    }

    .result-label {
        font-size: 0.85rem;
        color: #9f1239;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .result-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #881337;
        margin-top: 4px;
    }

    /* 6. Custom Falling Food Animation CSS */
    .falling-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 9999;
        overflow: hidden;
    }
    .food-item {
        position: absolute;
        top: -60px;
        font-size: 2.5rem;
        animation: fallDown 2.2s linear infinite;
    }
    @keyframes fallDown {
        0% { transform: translateY(0px) rotate(0deg); opacity: 1; }
        100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
    }
    </style>

    <!-- Side Floating Food Icons -->
    <div class="bg-icon" style="top: 12%; left: 4%;">🍒</div>
    <div class="bg-icon" style="top: 30%; right: 5%;">🥦</div>
    <div class="bg-icon" style="top: 52%; left: 3%;">🍎</div>
    <div class="bg-icon" style="top: 70%; right: 4%;">🍞</div>
    <div class="bg-icon" style="top: 86%; left: 5%;">🥛</div>
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
if "animating" not in st.session_state:
    st.session_state.animating = False

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
            <h2 style="font-size: 1.8rem; font-weight: 800; color: #881337; margin-bottom: 8px;">
                Hey, what are we cooking today? 🍳
            </h2>
            <p style="color: #be123c; font-size: 1rem; margin: 0;">
                Inspect your ingredients, check freshness, and plan your meals!
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Render Custom Falling Food Animation (Cherries, Broccoli, Apple, Bread, Milk, Egg)
    if st.session_state.animating:
        st.markdown(
            """
            <div class="falling-container">
                <div class="food-item" style="left: 10%; animation-delay: 0s;">🍒</div>
                <div class="food-item" style="left: 25%; animation-delay: 0.3s;">🥦</div>
                <div class="food-item" style="left: 40%; animation-delay: 0.1s;">🍎</div>
                <div class="food-item" style="left: 55%; animation-delay: 0.4s;">🍞</div>
                <div class="food-item" style="left: 70%; animation-delay: 0.2s;">🥛</div>
                <div class="food-item" style="left: 85%; animation-delay: 0.5s;">🥚</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Let's Go! 🚀", use_container_width=True):
            st.session_state.animating = True
            st.toast("🍒 🥦 🍎 Gathering your ingredients... 🍞 🥛 🥚")
            time.sleep(2.0)
            st.session_state.animating = False
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
                            <div class="result-val">{detected_food.title()}</div>
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
                        <div class="result-val" style="color: #be123c;">{max(0, estimated_expiry)} Days</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"Inference error: {e}")
