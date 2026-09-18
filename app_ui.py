import streamlit as st

# Custom Interface CSS
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

    /* Top Navigation Header Bar Fix */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 245, 247, 0.4) !important;
        backdrop-filter: blur(12px) !important;
    }
    header[data-testid="stHeader"] * {
        color: #881337 !important;
    }

    /* Override System Dark Mode Defaults for Widgets */
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

    /* Continuous Background Falling Food */
    .falling-container {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none; z-index: 999; overflow: hidden;
    }
    .food-item {
        position: absolute; 
        top: -80px; 
        font-size: 2.5rem;
        animation: fallDown 5s linear infinite;
    }
    
    @keyframes fallDown {
        0% { transform: translateY(-80px) rotate(0deg); opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(105vh) rotate(360deg); opacity: 0; }
    }
    </style>

    <!-- Falling Food Layer -->
    <div class="falling-container">
        <div class="food-item" style="left: 5%; animation-delay: 0s;">🍒</div>
        <div class="food-item" style="left: 15%; animation-delay: 1.5s;">🍊</div>
        <div class="food-item" style="left: 28%; animation-delay: 0.5s;">🍗</div>
        <div class="food-item" style="left: 40%; animation-delay: 2.2s;">🥚</div>
        <div class="food-item" style="left: 52%; animation-delay: 1.0s;">🥦</div>
        <div class="food-item" style="left: 65%; animation-delay: 2.8s;">🍎</div>
        <div class="food-item" style="left: 78%; animation-delay: 0.8s;">🥕</div>
        <div class="food-item" style="left: 90%; animation-delay: 2.0s;">🍇</div>
    </div>
""",
    unsafe_allow_html=True,
)
        )
