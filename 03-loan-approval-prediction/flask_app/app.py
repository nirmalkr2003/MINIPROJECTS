from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

# =====================================================
# APP INIT
# =====================================================
app = Flask(__name__)

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "loan_approval_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "model", "preprocessor.pkl")

# =====================================================
# LOAD MODEL & PREPROCESSOR
# =====================================================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

label_encoders = preprocessor["label_encoders"]
target_encoder = preprocessor["target_encoder"]
scaler = preprocessor["scaler"]
numerical_cols = preprocessor["numerical_cols"]
one_hot_cols = preprocessor["one_hot_cols"]
feature_order = preprocessor["feature_order"]

print("Model and preprocessor loaded successfully")

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------------------------------------------------
        # READ FORM DATA
        # -------------------------------------------------

        loan_term_years = float(request.form["Loan_Amount_Term_Years"])
        loan_term_months = loan_term_years * 12

        input_data = {
            "Gender": request.form["Gender"],
            "Married": request.form["Married"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "Dependents": int(request.form["Dependents"]),
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "Loan_Amount_Term": loan_term_months, 
            "Credit_History": int(request.form["Credit_History"]),
            "Property_Area": request.form["Property_Area"]
        }

        df = pd.DataFrame([input_data])

        # -------------------------------------------------
        # LABEL ENCODING
        # -------------------------------------------------
        for col, le in label_encoders.items():
            df[col] = le.transform(df[col])

        # -------------------------------------------------
        # ONE-HOT ENCODING
        # -------------------------------------------------
        df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

        # Ensure missing dummy columns exist
        for col in one_hot_cols:
            if col not in df.columns:
                df[col] = 0

        # -------------------------------------------------
        # LOG TRANSFORMATION
        # -------------------------------------------------
        df["ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
        df["CoapplicantIncome"] = np.log1p(df["CoapplicantIncome"])

        # -------------------------------------------------
        # SCALING
        # -------------------------------------------------
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # -------------------------------------------------
        # FEATURE ORDER ALIGNMENT
        # -------------------------------------------------
        df = df[feature_order]

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------
        prediction = model.predict(df)[0]
        result = target_encoder.inverse_transform([prediction])[0]

        if result == "Y":
            display_result = "Approved for Loan"
        else:
            display_result = "Not Eligible for Loan"


        return render_template(
            "index.html",
            prediction_text=f"Loan Approval Status: {display_result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
