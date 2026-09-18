"""
Smart Food Storage - Final App (Streamlit)
--------------------------------------------------
Combines all three upgraded models into one interface:
    1. Upload a food photo -> identifies the food (30-class image recognition model)
    2a. If it's apple/banana/carrot/tomato -> predicts freshness DIRECTLY from
        that same photo (photo-based freshness model)
    2b. For any other food -> falls back to entering days/storage details
        (the original Random Forest model, which covers all 30 foods)
    3. Shows real recipes matched from an 8000+ recipe dataset
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="PantryPal - Smart Food Storage", page_icon="🥑", layout="centered")

# ==================== BEAUTIFIED UI & INTERFACE CSS ====================
st.markdown(
    """
    <style>
    /* Force Light Color Scheme across System Dark/Light Modes */
    :root {
        color-scheme: light !important;
    }

    /* Animated Breathing Background */
    @keyframes pastelBreathe {
        0%   { background: radial-gradient(circle at 20% 20%, #ffe4e6 0%, #fff0f3 60%, #fff5f7 100%); }
        25%  { background: radial-gradient(circle at 80% 30%, #f3e8ff 0%, #fae8ff 60%, #fff5f7 100%); }
        50%  { background: radial-gradient(circle at 50% 80%, #dcfce7 0%, #f0fdf4 60%, #fff5f7 100%); }
        75%  { background: radial-gradient(circle at 20% 70%, #fce7f3 0%, #fee2e2 60%, #fff5f7 100%); }
        100% { background: radial-gradient(circle at 20% 20%, #ffe4e6 0%, #fff0f3 60%, #fff5f7 100%); }
    }

    .stApp {
        animation: pastelBreathe 20s ease-in-out infinite alternate;
        color: #4a041f !important;
        background-color: #fff5f7 !important;
    }

    /* Elevate Content Layer Above Background Elements */
    .stApp > header, .main, div[data-testid="stToolbar"] {
        z-index: 10 !important;
    }

    /* Top Navigation Header Bar Fix */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 245, 247, 0.4) !important;
        backdrop-filter: blur(12px) !important;
    }
    header[data-testid="stHeader"] * {
        color: #881337 !important;
    }

    /* Override System Dark Mode Defaults for Text & Labels */
    label, p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #4a041f !important;
    }

    /* Input Fields, Select Boxes, and Steppers */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input, select {
        background-color: #ffffff !important;
        color: #4a041f !important;
        border: 1px solid rgba(251, 113, 133, 0.4) !important;
        border-radius: 12px !important;
    }

    /* Dropdown Menus */
    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    ul[data-baseweb="menu"] li {
        color: #4a041f !important;
    }

    /* Brand Header Titles */
    .app-title-container {
        text-align: center;
        margin-top: 5px;
        margin-bottom: 20px;
    }
    .sub-brand {
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.25em;
        color: #9f1239 !important;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .main-brand {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #881337 0%, #be123c 50%, #fb7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(251, 113, 133, 0.3);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(136, 19, 55, 0.08);
        margin-bottom: 20px;
        text-align: center;
    }

    /* Landing Feature Cards */
    .landing-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-bottom: 25px;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(251, 113, 133, 0.3);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(136, 19, 55, 0.05);
    }
    .feature-icon { font-size: 1.6rem; margin-bottom: 4px; }
    .feature-title { font-size: 0.95rem; font-weight: 800; color: #881337 !important; }
    .feature-desc { font-size: 0.78rem; color: #be123c !important; margin-top: 2px; }

    /* Glowing Maroon Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #881337 0%, #be123c 50%, #f472b6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 12px 24px !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(136, 19, 55, 0.25), 0 0 15px rgba(244, 114, 182, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 12px 28px rgba(136, 19, 55, 0.35), 0 0 25px rgba(244, 114, 182, 0.5) !important;
    }

    /* Continuous Background Falling Food Animation Layer */
    .falling-container {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none !important; 
        z-index: 0 !important; 
        overflow: hidden;
    }
    .food-item {
        position: absolute; 
        top: -80px; 
        font-size: 2.5rem;
        pointer-events: none !important;
        animation: fallDown 6s linear infinite;
    }
    
    @keyframes fallDown {
        0% { transform: translateY(-80px) rotate(0deg); opacity: 0.8; }
        90% { opacity: 0.8; }
        100% { transform: translateY(105vh) rotate(360deg); opacity: 0; }
    }
    </style>

    <!-- Falling Food Overlay -->
    <div class="falling-container">
        <div class="food-item" style="left: 5%; animation-delay: 0s;">🍒</div>
        <div class="food-item" style="left: 15%; animation-delay: 1.8s;">🍊</div>
        <div class="food-item" style="left: 28%; animation-delay: 0.6s;">🍗</div>
        <div class="food-item" style="left: 40%; animation-delay: 2.5s;">🥚</div>
        <div class="food-item" style="left: 52%; animation-delay: 1.2s;">🥦</div>
        <div class="food-item" style="left: 65%; animation-delay: 3.0s;">🍎</div>
        <div class="food-item" style="left: 78%; animation-delay: 0.9s;">🥕</div>
        <div class="food-item" style="left: 90%; animation-delay: 2.2s;">🍇</div>
    </div>
""",
    unsafe_allow_html=True,
)

# Header Title Card
st.markdown(
    """
    <div class="app-title-container">
        <div class="sub-brand">SMART STORAGE AI</div>
        <div class="main-brand">PantryPal</div>
    </div>
    <div class="glass-card">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #881337 !important; margin-bottom: 4px;">
            Hey, what are we cooking today? 🔍
        </h2>
        <p style="color: #be123c !important; font-size: 0.95rem; margin: 0;">
            Inspect your ingredients, check freshness, and plan your meals!
        </p>
    </div>
    <div class="landing-grid">
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Identify</div>
            <div class="feature-desc">AI Photo Detection</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌱</div>
            <div class="feature-title">Freshness</div>
            <div class="feature-desc">Quality & Status</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⏳</div>
            <div class="feature-title">Expiry</div>
            <div class="feature-desc">Shelf Life Estimator</div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# =================================================================

PHOTO_FRESHNESS_FOODS = {
    "apple", "banana", "carrot", "tomato", "potato", "orange",
    "cucumber", "mango", "grape", "strawberry", "pepper",
    "meat", "bread",
}


def get_granular_freshness(prediction, confidence):
    """
    Turns the binary fresh/not_fresh prediction + confidence score into a
    more informative 6-level freshness scale, using confidence as a proxy
    for how strongly the photo matches "clearly fresh" vs "clearly spoiled".
    """
    if prediction == "fresh":
        if confidence >= 90:
            return "🟢", "Very Fresh"
        elif confidence >= 70:
            return "🟢", "Fresh"
        else:
            return "🟡", "Slightly Fresh (borderline)"
    else:  # not_fresh
        if confidence >= 90:
            return "🔴", "Heavily Spoiled"
        elif confidence >= 70:
            return "🟠", "Spoiled"
        else:
            return "🟡", "Starting to Spoil"

STATUS_DISPLAY = {
    "fresh": ("✅", "Fresh"),
    "expiring_soon": ("⚠️", "Expiring Soon"),
    "spoiled": ("❌", "Spoiled"),
}


# ---------- Load everything once, cached ----------
@st.cache_resource
def load_image_model():
    model = tf.keras.models.load_model("food_model.keras")
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


@st.cache_resource
def load_freshness_photo_model():
    model = tf.keras.models.load_model("freshness_model_v2.keras")
    with open("freshness_v2_class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


@st.cache_resource
def load_expiry_model():
    model = joblib.load("expiry_model.joblib")
    food_encoder = joblib.load("food_encoder.joblib")
    storage_encoder = joblib.load("storage_encoder.joblib")
    status_encoder = joblib.load("status_encoder.joblib")
    return model, food_encoder, storage_encoder, status_encoder


@st.cache_resource
def load_recipes():
    with open("recipe_lookup.json") as f:
        return json.load(f)


try:
    image_model, class_names = load_image_model()
    freshness_photo_model, freshness_class_names = load_freshness_photo_model()
    expiry_model, food_encoder, storage_encoder, status_encoder = load_expiry_model()
    recipe_lookup = load_recipes()
    models_loaded = True
except Exception as e:
    st.error(f"Couldn't load model files: {e}")
    models_loaded = False


def show_recipes(food_name, prediction):
    """Displays recipe suggestions for a food, or a discard warning if spoiled.
    Handles both the Random Forest's labels (fresh/expiring_soon/spoiled)
    and the photo model's binary labels (fresh/not_fresh)."""
    if prediction in ("spoiled", "not_fresh"):
        st.warning(f"This {food_name} is likely spoiled — consider discarding it.")
        return

    recipes = recipe_lookup.get(food_name, [])
    if prediction == "expiring_soon":
        st.info(f"Your {food_name} is expiring soon — here are some recipes to use it up:")
    else:
        st.write(f"Here are some recipe ideas for your {food_name}:")

    if recipes:
        for recipe in recipes:
            with st.expander(f"🍳 {recipe['name']} (⭐ {recipe['rating']})"):
                st.write(recipe["instructions"])
                if recipe.get("prep_time") or recipe.get("cook_time"):
                    st.caption(f"Prep: {recipe.get('prep_time', '?')} | Cook: {recipe.get('cook_time', '?')}")
    else:
        st.write("No recipes found for this food yet.")


if models_loaded:
    # ---------- Step 1: Identify the food ----------
    st.header("Step 1: Identify the food")
    identified_food = None
    img = None

    input_method = st.radio("How would you like to provide a photo?", ["📁 Upload a photo", "📷 Use camera"], horizontal=True)

    uploaded_photo = None
    if input_method == "📁 Upload a photo":
        uploaded_photo = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])
    else:
        uploaded_photo = st.camera_input("Take a photo of the food")

    if uploaded_photo is not None:
        img = Image.open(uploaded_photo).convert("RGB")
        st.image(img, caption="Photo", width=250)

        img_resized = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = image_model.predict(img_array)
        scores = predictions[0]  # model already outputs softmax probabilities
        identified_food = class_names[np.argmax(scores)]
        confidence = 100 * np.max(scores)

        st.success(f"Identified: **{identified_food}** ({confidence:.1f}% confidence)")

    # ---------- Step 2: Freshness ----------
    st.header("Step 2: Check freshness")

    if identified_food in PHOTO_FRESHNESS_FOODS and img is not None:
        # Use the photo-based freshness model directly on the same uploaded photo
        st.write(f"Checking freshness directly from the photo (supported for {identified_food})...")

        if st.button("Predict freshness from photo"):
            fresh_predictions = freshness_photo_model.predict(img_array)
            fresh_scores = fresh_predictions[0]  # model already outputs softmax probabilities
            prediction = freshness_class_names[np.argmax(fresh_scores)]
            confidence = 100 * np.max(fresh_scores)

            icon, label = get_granular_freshness(prediction, confidence)
            st.markdown(f"### {icon} Status: {label}")
            st.write(f"Confidence: {confidence:.1f}%")

            show_recipes(identified_food, prediction)

    else:
        # Fall back to the Random Forest model with manually entered details
        if identified_food is not None:
            st.write(f"Photo-based freshness isn't available for **{identified_food}** yet — enter storage details instead:")

        available_foods = list(food_encoder.classes_)
        default_index = available_foods.index(identified_food) if identified_food in available_foods else 0

        food_choice = st.selectbox("Food item", available_foods, index=default_index)
        days_since_purchase = st.number_input("Days since purchase", min_value=0.0, value=3.0, step=1.0)
        storage_choice = st.selectbox("Storage type", list(storage_encoder.classes_))
        temperature_c = st.slider("Temperature (°C)", min_value=0.0, max_value=35.0, value=20.0)
        humidity_level = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

        if st.button("Predict freshness"):
            food_encoded = food_encoder.transform([food_choice])[0]
            storage_encoded = storage_encoder.transform([storage_choice])[0]

            input_data = pd.DataFrame([{
                "food_item_encoded": food_encoded,
                "days_since_purchase": days_since_purchase,
                "storage_type_encoded": storage_encoded,
                "temperature_c": temperature_c,
                "humidity_level": humidity_level,
            }])

            prediction_encoded = expiry_model.predict(input_data)[0]
            prediction = status_encoder.inverse_transform([prediction_encoded])[0]
            probabilities = expiry_model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100

            icon, label = STATUS_DISPLAY.get(prediction, ("", prediction))
            st.markdown(f"### {icon} Status: {label}")
            st.write(f"Confidence: {confidence:.1f}%")

            show_recipes(food_choice, prediction)
