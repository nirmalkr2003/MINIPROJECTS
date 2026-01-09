import streamlit as st
import numpy as np
import pickle
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

with open(os.path.join(MODEL_DIR, 'gbr_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaling.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# =====================================================
# HEADER
# =====================================================
st.markdown("<h1 style='text-align:center;'>🏠 Boston Housing Price Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Estimate the median house value based on neighborhood characteristics</p>",
    unsafe_allow_html=True
)
st.divider()

# =====================================================
# INPUT FORM (USER FRIENDLY)
# =====================================================
with st.form("boston_form"):
    st.subheader("📋 Enter Property & Area Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        crim = st.number_input(
            "Crime Rate in the Area",
            help="Per capita crime rate by town",
            min_value=0.0,
            step=0.01
        )

        zn = st.number_input(
            "Residential Land Zoning (%)",
            help="Percentage of residential land zoned for large plots",
            min_value=0.0,
            max_value=100.0
        )

        chas = st.selectbox(
            "Near Charles River?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        nox = st.number_input(
            "Air Pollution Level (NOX)",
            help="Nitric oxide concentration",
            min_value=0.0,
            step=0.01
        )

    with col2:
        rm = st.number_input(
            "Average Number of Rooms",
            help="Average rooms per dwelling",
            min_value=1.0,
            step=0.1
        )

        age = st.number_input(
            "Age of Houses (%)",
            help="Percentage of owner-occupied units built before 1940",
            min_value=0.0,
            max_value=100.0
        )

        dis = st.number_input(
            "Distance to Employment Centers",
            help="Weighted distance to Boston employment hubs",
            min_value=0.0,
            step=0.1
        )

        rad = st.number_input(
            "Highway Accessibility Index",
            help="Index of accessibility to radial highways",
            min_value=1,
            step=1
        )

    with col3:
        tax = st.number_input(
            "Property Tax Rate",
            help="Full-value property tax rate per $10,000",
            min_value=0.0,
            step=1.0
        )

        ptratio = st.number_input(
            "Student–Teacher Ratio",
            help="Pupil–teacher ratio by town",
            min_value=1.0,
            step=0.1
        )

        b = st.number_input(
            "Diversity Index (B)",
            help="1000(Bk − 0.63)² where Bk is proportion of Black residents",
            min_value=0.0,
            step=1.0
        )

        lstat = st.number_input(
            "Lower Status Population (%)",
            help="Percentage of lower socio-economic population",
            min_value=0.0,
            max_value=100.0
        )

    submitted = st.form_submit_button("🔍 Predict House Price")

# =====================================================
# PREDICTION
# =====================================================
if submitted:
    # Feature engineering (same as training)
    rooms_per_age = rm / (age + 1)
    tax_per_room = tax / rm

    input_data = np.array([[
        crim, zn, chas, nox, rm, age, dis, rad,
        tax, ptratio, b, lstat,
        rooms_per_age, tax_per_room
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.divider()
    st.success(f"Estimated Median House Value: **${prediction:.2f} (in $1000s)**")
