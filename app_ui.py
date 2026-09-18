import os
import time
import joblib
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Page Config
st.set_page_config(page_title="PantryPal", page_icon="🥑", layout="centered")

# Custom CSS for Light Baby Pink, Pastel Breathing, Falling Food, and Maroon Buttons
st.markdown(
    """
    <style>
    /* Base Background and Breathing Glow */
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

    /* Floating Side Icons */
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

    /* Titles */
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

    /* Feature Cubes Grid on Landing */
    .landing-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-bottom: 25px;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(251, 113, 133, 0.3);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(136, 19, 55, 0.05);
    }
    .feature-icon { font-size: 1.6rem; margin-bottom: 4px; }
    .feature-title { font-size: 0.95rem; font-weight: 800; color: #881337; }
    .feature-desc { font-size: 0.78rem; color: #be123c; margin-top: 2px; }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(251, 113, 133, 0.3);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(136, 19, 55, 0.08);
        margin-bottom: 20px;
    }

    /* Falling Food Overlay */
    .falling-container {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9999; overflow: hidden;
    }
    .food-item {
        position: absolute; top: -60px; font-size: 2.6rem;
        animation: fallDown 2.2s linear infinite;
    }
    @keyframes fallDown {
        0% { transform: translateY(0px) rotate(0deg); opacity: 1; }
        100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
    }
    </style>

    <!-- Side Floating Background Icons -->
    <div class="bg-icon" style="top: 10%; left: 3%;">🍒</div>
    <div class="bg-icon" style="top: 25%; right: 4%;">🥦</div>
    <div class="bg-icon" style="top: 45%; left: 2%;">🍎</div>
    <div class="bg-icon" style="top: 65%; right: 3%;">🍇</div>
    <div class="bg-icon" style="top: 85%; left: 4%;">🥕</div>
""",
    unsafe_allow_html=True,
)


# Load Trained Models & Encoders
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

# Navigation & Animation State
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
        <div class="glass-card" style="text-align: center; margin-bottom: 25px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: #881337; margin-bottom: 6px;">
                Hey, what are we cooking today? 🍳
            </h2>
            <p style="color: #be123c; font-size: 1rem; margin: 0;">
                Inspect your ingredients, check freshness, and plan your meals!
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # 3 Feature Cubes
    st.markdown(
        """
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

    # Trigger Custom Falling Foods (Cherries, Oranges, Chicken, Egg, Milk, Broccoli, Apple, Grapes, Carrot, Cucumber, Eggplant, Strawberries, Peppers)
    if st.session_state.animating:
        st.markdown(
            """
            <div class="falling-container">
                <div class="food-item" style="left: 5%; animation-delay: 0s;">🍒</div>
                <div class="food-item" style="left: 12%; animation-delay: 0.3s;">🍊</div>
                <div class="food-item" style="left: 20%; animation-delay: 0.1s;">🍗</div>
                <div class="food-item" style="left: 28%; animation-delay: 0.4s;">🥚</div>
                <div class="food-item" style="left: 36%; animation-delay: 0.2s;">🥛</div>
                <div class="food-item" style="left: 44%; animation-delay: 0.5s;">🥦</div>
                <div class="food-item" style="left: 52%; animation-delay: 0.1s;">🍎</div>
                <div class="food-item" style="left: 60%; animation-delay: 0.3s;">🍇</div>
                <div class="food-item" style="left: 68%; animation-delay: 0.0s;">🥕</div>
                <div class="food-item" style="left: 76%; animation-delay: 0.4s;">🥒</div>
                <div class="food-item" style="left: 84%; animation-delay: 0.2s;">🍆</div>
                <div class="food-item" style="left: 92%; animation-delay: 0.5s;">🍓</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Let's Go! 🚀", use_container_width=True):
            st.session_state.animating = True
            st.toast("🍒 🍊 🍗 🥚 🥛 🥦 🍎 Gathering ingredients...")
            time.sleep(2.0)
            st.session_state.animating = False
            st.session_state.page = "app"
            st.rerun()

# ================= PAGE 2: FULL MAIN INSPECTION APP =================
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

    # SECTION 1: PHOTO UPLOAD OR CAMERA CAPTURE
    st.markdown("### Upload or Take a Photo of Ingredient")

    tab_upload, tab_camera = st.tabs(
        ["📁 Upload Photo", "📷 Take Photo of Ingredient"]
    )
    uploaded_file = None

    with tab_upload:
        file_up = st.file_uploader(
            "Upload food image...", type=["jpg", "jpeg", "png"]
        )
        if file_up:
            uploaded_file = file_up

    with tab_camera:
        cam_up = st.camera_input("Take a photo of ingredient")
        if cam_up:
            uploaded_file = cam_up

    detected_food_name = "apple"  # Default fallback

    if uploaded_file is not None:
        st.image(
            uploaded_file, caption="Uploaded photo", width=300
        )  # Matches original sizing[cite: 11]

        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_resized = ImageOps.fit(
                image, (224, 224), Image.Resampling.LANCZOS
            )
            img_array = np.asarray(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Food Prediction + Confidence Score
            food_preds = assets["food_model"].predict(img_array)
            top_idx = np.argmax(food_preds[0])
            detected_food_name = assets["food_classes"][top_idx]
            confidence_pct = float(food_preds[0][top_idx]) * 100

            # Matches original confidence bar
            st.success(
                f"**Identified: {detected_food_name} ({confidence_pct:.1f}% confidence)**"
            )

            # Direct photo freshness prediction if button pressed
            st.markdown(
                f"*Checking freshness directly from the photo (supported for {detected_food_name})...*"
            )
            if st.button("Predict freshness from photo"):
                freshness_preds = assets["freshness_model"].predict(img_array)
                fresh_idx = np.argmax(freshness_preds[0])
                fresh_status = assets["freshness_classes"][fresh_idx]
                fresh_conf = float(freshness_preds[0][fresh_idx]) * 100

                # Render Status & Recipes
                st.markdown(
                    f"### ✅ Status: {fresh_status.title()} \n**Confidence: {fresh_conf:.1f}%**"
                )

        except Exception as e:
            st.error(f"Inference error: {e}")

    st.markdown("---")

    # SECTION 2: CHECK FRESHNESS & MANUAL CONTROLS (From Screenshot 4)
    st.markdown("## Step 2: Check freshness")

    food_options = (
        assets.get("food_classes")
        if assets.get("food_classes")
        else ["apple", "carrot", "orange", "banana"]
    )
    default_index = (
        food_options.index(detected_food_name)
        if detected_food_name in food_options
        else 0
    )

    selected_food = st.selectbox("Food item", food_options, index=default_index)
    days_since = st.number_input(
        "Days since purchase", min_value=0.0, max_value=60.0, value=3.0, step=1.0
    )
    storage_type = st.selectbox(
        "Storage type", ["fridge", "pantry", "freezer"]
    )  # Fixed key 'fridge'!
    temp_val = st.slider(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=40.0,
        value=20.0,
        step=0.5,
    )
    humidity_val = st.slider(
        "Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0
    )

    if st.button("Predict freshness"):
        try:
            # Transform inputs safely using encoders
            food_enc = (
                assets["food_encoder"].transform([selected_food])[0]
                if "food_encoder" in assets
                else 0
            )
            storage_enc = (
                assets["storage_encoder"].transform([storage_type])[0]
                if "storage_encoder" in assets
                else 0
            )
            status_dummy = 0

            # 5-Feature Vector matching expiry model expectation: [food, status, storage, temp, humidity]
            feat_arr = np.array(
                [[food_enc, status_dummy, storage_enc, temp_val, humidity_val]]
            )
            est_days = assets["expiry_model"].predict(feat_arr)[0]

            st.markdown(
                f"""
                <div class="glass-card">
                    <h3 style="color:#881337; margin:0;">Estimated Remaining Freshness</h3>
                    <p style="font-size:2rem; font-weight:800; color:#be123c; margin:5px 0 0 0;">
                        {max(0, int(est_days))} Days Remaining
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        except Exception as err:
            st.error(f"Prediction error: {err}")

    # SECTION 3: RECIPES SECTION (Matches Screenshots 5 & 6)[cite: 13, 14]
    st.markdown("---")
    st.markdown(f"### Here are some recipe ideas for your {selected_food}:")

    # Sample Expandable Recipes
    with st.expander(f"🍳 Savory Shrimps On Skillet Recipe (⭐ 5.0)"):
        st.write(
            "To begin making the Savory Shrimps On Skillet recipe, devein and deshell the shrimps."
            " Add 3-4 drops of orange or lemon food color/juice into the marinated shrimps and mix well."
            " Heat a skillet with butter and cook to perfection!"
        )
        st.caption("Prep: 20 M | Cook: 8 M")

    with st.expander("🍳 Thandai Nectarine Mini Galette Recipe (⭐ 5.0)"):
        st.write(
            "A delicious dessert pie made with fresh fruit, ground nuts, and spices baked in a pastry crust."
        )
        st.caption("Prep: 30 M | Cook: 25 M")

    with st.expander("🍳 Marrakesh Vegetable Curry Recipe (⭐ 5.0)"):
        st.write(
            "A rich, aromatic vegetable curry loaded with carrots, apples, chickpeas, and warm spices."
        )
        st.caption("Prep: 15 M | Cook: 30 M")
