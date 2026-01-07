# Car Price Prediction System

## Project Overview
This project predicts the selling price of a used car using machine learning based on features such as year, fuel type, transmission, and kilometers driven.

The project provides:
- A Flask web application
- A Streamlit interactive interface

---

## Project Structure

flask_project/
│
├── data/
│   └── car data.csv
│
├── training/
│   └── model_training.ipynb
│
├── flask_app/
│   ├── app.py
│   ├── model/
│   │   ├── model.pkl
│   │   ├── scaling.pkl
│   │   ├── Fuel_Type.pkl
│   │   └── Transmission.pkl
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── images/
│           └── bg.jpg
│
├── streamlit_app/
│   └── main.py
│
├── requirements.txt
└── README.md

---

## Requirements

Install dependencies using:

pip install -r requirements.txt

---

## Run Flask Application

cd flask_app  
python app.py  

Open in browser:
http://127.0.0.1:5000

---

## Run Streamlit Application

cd streamlit_app  
streamlit run main.py

---

## Model Details
- Algorithm: Regression (scikit-learn)
- Model stored using pickle

