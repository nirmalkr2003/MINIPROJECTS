# 🍷 Wine Type Prediction Dashboard

A machine learning–based **Streamlit web application** that predicts whether a wine is **Red** or **White** using its chemical properties.  
The project demonstrates a complete **end-to-end ML workflow** with a **professionally styled interactive dashboard**.

---

## Project Overview

This application uses a **Random Forest Classifier** trained on wine chemical composition data to classify wine type.  
Users can input physicochemical attributes of wine and instantly receive predictions via a clean, dashboard-style UI.

---

## Machine Learning Details

- **Problem Type:** Binary Classification  
- **Target Variable:** `type`
  - `Red`
  - `White`
- **Algorithm Used:** Random Forest Classifier  
- **Why Random Forest?**
  - Handles non-linear feature interactions
  - Robust to noise and outliers
  - Performs well on tabular chemical data
  - No need for feature scaling

---

## Features Used for Prediction

- Fixed Acidity  
- Volatile Acidity  
- Citric Acid  
- Residual Sugar  
- Chlorides  
- Free Sulfur Dioxide  
- Total Sulfur Dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol (%)  
- Quality Score  

---

## Application Features

- Three-column dashboard layout
- Grouped input sections:
  - **Acidity & Sugar**
  - **Sulfur & Density**
  - **Taste & Alcohol**
- Custom CSS styling (white main area, dark sidebar)
- Interactive numeric inputs
- Real-time prediction output
- Clean and professional UI suitable for demos, viva, and interviews

---

## Project Structure

wine_type_classifier/
│
├── data/
│ └── wine.csv
│
├── model/
│ ├── wine_type_model.pkl
│ └── wine_type_encoder.pkl
│
├── streamlit_app/
│ └── app.py
│
├── requirements.txt
└── README.md


---

## ▶ How to Run the Project

### Clone the repository
```bash
git clone <your-repo-url>
cd wine_type_classifier
```

## Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / Mac
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run Streamlit
```bash
cd streamlit_app
streamlit run app.py





