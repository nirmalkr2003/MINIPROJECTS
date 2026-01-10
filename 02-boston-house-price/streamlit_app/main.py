import streamlit as st
import numpy as np
import pickle
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Boston Housing Valuation",
    page_icon="🏠",
    layout="centered"
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

model = pickle.load(open(os.path.join(MODEL_DIR, 'gbr_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaling.pkl'), 'rb'))

# =====================================================
# SIDEBAR — EXPERT CONTEXT
# =====================================================
with st.sidebar:
    st.markdown("## 🏠 Boston Housing Valuation")
    st.markdown(
        """
        This  tool estimates the **median house  value** in  Boston
        using  **socio-economic, structural and environmental indicators**.

        ---
        ### 🔍 How to Use
        1. Enter neighborhood characteristics  
        2. Provide property-level details  
        3. Click **Estimate Property Value**

        ---
        ### 🧠 Model Insights
        - Non-linear regression model  
        - Handles complex feature interactions  
        - Trained on historical Boston housing data  

        ---
        ### ⚠️ Note
        Predictions are **estimates**, not financial advice.
        """
    )

# =====================================================
# CUSTOM CSS (SUBTLE, PROFESSIONAL)
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
    }

    h1 {
        color: #111827;
        font-weight: 700;
        letter-spacing: -0.4px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 17px;
        font-weight: 600;
        color: #1f2933;
        margin-top: 25px;
    }

    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 25px 0;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        height: 42px;
    }

    div[data-testid="stNumberInput"] p {
        color: #111827;
        font-weight: 500;
    }

    div.stButton > button {
        background-color: #111827;
        color: white !important;
        border-radius: 6px;
        height: 46px;
        font-size: 15px;
        font-weight: 600;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1f2933;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
   .hero-box {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 16px;   /* ↓ reduced spacing */
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}


    .hero-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 10px;
    }

    .hero-meta {
        font-size: 13px;
        color: #374151;
    }

    .hero-meta span {
        margin-right: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🏠 Boston Housing Valuation</div>
        <div class="hero-subtitle">
            Machine-learning based estimation of median house values
        </div>
        <div class="hero-meta">
            <span><b>Model:</b> Gradient Boosting Regressor</span>
            <span><b>Data:</b> Boston Housing Dataset</span>
            <span><b>Output:</b> Price (in $1000s)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Fix number input text visibility */
    div[data-testid="stNumberInput"] input {
        color: #111827 !important;          /* visible dark text */
        caret-color: #111827 !important;    /* cursor color */
    }

    /* Fix selectbox text */
    div[data-testid="stSelectbox"] div {
        color: #111827 !important;
    }

    /* Placeholder text */
    div[data-testid="stNumberInput"] input::placeholder {
        color: #9ca3af !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Make selectbox background white */
    div[data-testid="stSelectbox"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        min-height: 42px;
    }

    /* Text inside selectbox */
    div[data-testid="stSelectbox"] div[role="combobox"] {
        color: #111827 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Selectbox container */
    div[data-testid="stSelectbox"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
    }

    /* Selected value text */
    div[data-testid="stSelectbox"] span {
        color: #111827 !important;
    }

    /* Dropdown menu */
    div[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }

    /* Dropdown options */
    div[role="option"] {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* Hovered option */
    div[role="option"]:hover {
        background-color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# =====================================================
# MAIN HEADER
# =====================================================
st.markdown("<h1>Boston Housing Valuation</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict median house value using socio-economic and structural indicators</div>",
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# =====================================================
# INPUT SECTIONS
# =====================================================

st.markdown("<div class='section-title'>Neighborhood Characteristics</div>", unsafe_allow_html=True)

crim = st.number_input("Crime Rate", min_value=0.0, step=0.01)
zn = st.number_input("Residential Zoning (%)", min_value=0.0, max_value=100.0)
chas = st.selectbox("Near Charles River", [0, 1], format_func=lambda x: "Yes" if x else "No")
nox = st.number_input("Air Pollution (NOX)", min_value=0.0, step=0.01)

st.markdown("<div class='section-title'>Property Attributes</div>", unsafe_allow_html=True)

rm = st.number_input("Average Rooms per Dwelling", min_value=1.0, step=0.1)
age = st.number_input("Age of Property (%)", min_value=0.0, max_value=100.0)
tax = st.number_input("Property Tax Rate", min_value=0.0)
ptratio = st.number_input("Student–Teacher Ratio", min_value=1.0)

st.markdown("<div class='section-title'>Accessibility & Socio-Economic Factors</div>", unsafe_allow_html=True)

dis = st.number_input("Distance to Employment Centers", min_value=0.0, step=0.1)
rad = st.number_input("Highway Accessibility Index", min_value=1, step=1)
b = st.number_input("Diversity Index (B)", min_value=0.0)
lstat = st.number_input("Lower Status Population (%)", min_value=0.0, max_value=100.0)

st.markdown("<hr>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================
if st.button("Estimate Property Value"):
    rooms_per_age = rm / (age + 1)
    tax_per_room = tax / rm

    input_data = np.array([[ 
        crim, zn, chas, nox, rm, age, dis, rad,
        tax, ptratio, b, lstat,
        rooms_per_age, tax_per_room
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Median House Value: **${prediction:.2f} (in $1000s)**")
