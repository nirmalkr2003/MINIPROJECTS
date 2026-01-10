import streamlit as st
import numpy as np
import pickle
import os

st.markdown(
    """
    <style>
    /* =========================
       MAIN PAGE (CENTER AREA)
    ========================== */
    .stApp {
        background-color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        color: white;
    }

    /* =========================
       MAIN CONTENT CONTAINER
    ========================== */
    section.main > div {
        background-color: #ffffff;
    }

    /* =========================
       INPUT TITLES (Fixed Acidity etc.)
    ========================== */
    div[data-testid="stNumberInput"] p {
        color: #000000 !important;
        font-weight: 500;
        opacity: 1 !important;
    }

    div[data-testid="stSelectbox"] p {
        color: #000000 !important;
        font-weight: 500;
    }

    /* =========================
       NUMBER INPUT BOX
    ========================== */
    div[data-testid="stNumberInput"] input {
        background-color: #1f2933;
        color: white;
        border-radius: 10px;
        border: none;
        height: 44px;
    }

    /* =========================
       BUTTON STYLE
    ========================== */
    div.stButton > button {
        background: linear-gradient(135deg, #111827, #1f2933);
        color: white !important;
        border-radius: 10px;
        height: 48px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #e5e7eb, #f3f4f6);
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Main title (st.title) */
    h1 {
        color: #111827 !important;   /* dark slate */
        font-weight: 700;
        opacity: 1 !important;
    }

    /* Subheading text under title */
    h1 + div p {
        color: #374151 !important;   /* dark gray */
        font-size: 16px;
        font-weight: 500;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Subheaders like "Acidity & Sugar", "Sulfur & Density" */
    h3 {
        color: #111827 !important;   /* dark slate / near black */
        font-weight: 600;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)





# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Wine Type Classifier",
    page_icon="🍷",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

model = pickle.load(open(os.path.join(MODEL_DIR, "wine_type_model.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(MODEL_DIR, "wine_type_encoder.pkl"), "rb"))

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.title("About Project")
    st.markdown(
        """
        **Wine Type Prediction System**

        This application classifies wine as **Red** or **White**
        using a trained **Random Forest Classifier**.

        **Features Used:**
        - Chemical properties
        - Alcohol content
        - Density & acidity
        """
    )

# =====================================================
# MAIN HEADER
# =====================================================
st.markdown("<h1 style='text-align:center;'>Wine Type Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Enter chemical properties of wine</p>", unsafe_allow_html=True)

st.divider()

# =====================================================
# INPUT SECTIONS (3 COLUMNS)
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Acidity & Sugar")
    fixed_acidity = st.number_input("Fixed Acidity", 0.0, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", 0.0, step=0.01)
    citric_acid = st.number_input("Citric Acid", 0.0, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", 0.0, step=0.1)

with col2:
    st.subheader("Sulfur & Density")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0)
    chlorides = st.number_input("Chlorides", 0.0, step=0.01)
    density = st.number_input("Density", 0.0, step=0.01)

with col3:
    st.subheader("Taste & Alcohol")
    pH = st.number_input("pH", 0.0, step=0.01)
    sulphates = st.number_input("Sulphates", 0.0, step=0.01)
    alcohol = st.number_input("Alcohol (%)", 0.0, step=0.1)
    quality = st.number_input("Quality Score", 0, 10, step=1)

st.divider()

# =====================================================
# PREDICT BUTTON
# =====================================================
center = st.columns([3, 2, 3])
with center[1]:
    predict = st.button("🔍 Predict Wine Type", use_container_width=True)

# =====================================================
# PREDICTION LOGIC
# =====================================================
if predict:
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, quality
    ]])

    prediction = model.predict(input_data)[0]
    result = encoder.inverse_transform([prediction])[0]

    st.divider()

    if result.lower() == "white":
        st.success("**This wine is classified as WHITE WINE**")
    else:
        st.error("**This wine is classified as RED WINE**")

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<hr><p style='text-align:center;color:gray;'>ML Dashboard | Wine Classification</p>",
    unsafe_allow_html=True
)
