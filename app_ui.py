import os
import joblib
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Page Configuration
st.set_page_config(
    page_title="PantryPalSmart | AI Food Storage",
    page_icon="🥑",
    layout="centered"
)

# Dark Glassmorphism Styling
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 20%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        color: #f8fafc;
    }
    
    .badge {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
        display: inline-block;
        margin-bottom: 12px;
    }
    
    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 20px;
        margin-top: 15px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .result-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .result-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #34d399;
    }
    </style>
""", unsafe_allow_html=True)


# Load Models and Helper Files safely
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

    except Exception as e:
        st.error(f"Error loading model files: {e}")
    return assets


assets = load_assets()

# Main Application Header
st.markdown('<div class="badge">PANTRYPAL SMART</div>', unsafe_allow_html=True)
st.title("Smart Food Storage AI")
st.caption(
    "Instantly identify food items, inspect freshness levels, and calculate"
    " shelf life."
)

# Image Upload Interface
uploaded_file = st.file_uploader(
    "Choose a food photo...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display preview with updated Streamlit keyword
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Inspect Item", type="primary"):
        with st.spinner("Analyzing item with AI models..."):
            try:
                # Preprocess image
                image = Image.open(uploaded_file).convert("RGB")
                image_resized = ImageOps.fit(
                    image, (224, 224), Image.Resampling.LANCZOS
                )
                img_array = np.asarray(image_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict Food Classification
                food_preds = assets["food_model"].predict(img_array)
                food_idx = np.argmax(food_preds[0])
                detected_food = assets["food_classes"][food_idx]

                # Predict Freshness
                freshness_preds = assets["freshness_model"].predict(img_array)
                freshness_idx = np.argmax(freshness_preds[0])
                freshness_status = assets["freshness_classes"][freshness_idx]

                # Calculate Shelf Life Expiry Prediction
                if "food_encoder" in assets and "status_encoder" in assets:
                    food_enc = assets["food_encoder"].transform([detected_food])[
                        0
                    ]
                    status_enc = assets["status_encoder"].transform(
                        [freshness_status]
                    )[0]
                    features = np.array([[food_enc, status_enc]])
                    estimated_expiry = int(
                        assets["expiry_model"].predict(features)[0]
                    )
                else:
                    estimated_expiry = 5  # Fallback estimate

                # Output Display Cards
                st.markdown("### Analysis Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                    <div class="glass-card">
                        <div class="result-label">Food Detected</div>
                        <div class="result-val" style="color: #FFF;">{detected_food.title()}</div>
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
                st.error(
                    f"Error processing image or model inference failed: {e}"
                )
