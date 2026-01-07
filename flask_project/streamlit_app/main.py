import streamlit as st
import pickle
import numpy as np
import os

# ---------- PAGE STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------- BACKGROUND ----------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1920&q=80
            ");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
""", unsafe_allow_html=True)

# ---------- TEXT & UI STYLING ----------
st.markdown("""
<style>
h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 42px !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,1);
}
p {
    color: #f1f1f1 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    text-shadow: 1px 1px 6px rgba(0,0,0,0.8);
}
label {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.9);
}
input, select {
    background-color: rgba(20, 20, 20, 0.85) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] > div {
    background-color: rgba(20, 20, 20, 0.85) !important;
    color: white !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #00e676, #00c853) !important;
    color: #000000 !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
}
div[data-testid="stAlert"] {
    background-color: rgba(0, 0, 0, 0.7) !important;
    color: #00e676 !important;
    font-weight: 700 !important;
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '..', 'model', 'model.pkl')
fuel_path  = os.path.join(BASE_DIR, '..', 'model', 'Fuel_Type.pkl')
trans_path = os.path.join(BASE_DIR, '..', 'model', 'Transmission.pkl')
scale_path = os.path.join(BASE_DIR, '..', 'model', 'scaling.pkl')

model = pickle.load(open(model_path, 'rb'))
Fuel_Type_en = pickle.load(open(fuel_path, 'rb'))
Transmission_en = pickle.load(open(trans_path, 'rb'))
scaler = pickle.load(open(scale_path, 'rb'))
print("Model and encoders loaded")
# model = pickle.load(open('model.pkl', 'rb'))
# Fuel_Type_en = pickle.load(open('Fuel_Type.pkl', 'rb'))
# Transmission_en = pickle.load(open('Transmission.pkl', 'rb'))
# scaler = pickle.load(open('scaling.pkl', 'rb'))

st.set_page_config(page_title="Car Price Prediction")

# ---------- HOME PAGE ----------
def home_page():
    st.title("Car Price Prediction App")
    st.write("Enter the car details below to estimate its selling price:")

    col1, col2 = st.columns(2)

    with col1:
        Year = st.text_input("Year of Manufacture (e.g. 2015)", "2015")
        Present_Price = st.text_input("Present Price (in lakhs)", "5.0")
        Kms_Driven = st.text_input("Kilometers Driven", "30000")

    with col2:
        Fuel_Type = st.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG"))
        Transmission = st.selectbox("Transmission", ("Manual", "Automatic"))
        Owner = st.text_input("Number of Previous Owners", "0")
        Seller_Type_Individual = st.selectbox("Seller Type", ("Dealer", "Individual"))

    if st.button("Predict Car Price"):
        try:
            Year = int(Year)
            Present_Price = float(Present_Price)
            Kms_Driven = float(Kms_Driven)
            Owner = int(Owner)

            Fuel_Type_val = Fuel_Type_en.transform([Fuel_Type])[0]
            Transmission_val = Transmission_en.transform([Transmission])[0]
            Seller_Type_Individual_val = 1 if Seller_Type_Individual == "Individual" else 0

            details = [
                Year, Present_Price, Kms_Driven,
                Fuel_Type_val, Transmission_val,
                Owner, Seller_Type_Individual_val
            ]

            data_out = np.array(details).reshape(1, -1)
            data_scaled = scaler.transform(data_out)

            prediction = model.predict(data_scaled)[0]

            st.session_state.prediction = round(prediction, 2)
            st.session_state.page = "result"

        except ValueError:
            st.warning("Please enter valid numeric values.")

# ---------- RESULT PAGE ----------
def result_page():
    st.title("Prediction Result ")
    st.success(f"Estimated Car Price: ₹ {st.session_state.prediction} Lakhs")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

# ---------- PAGE ROUTING ----------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "result":
    result_page()
