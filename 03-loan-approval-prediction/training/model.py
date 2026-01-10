import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "loan.csv")

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)

# =====================================================
# MISSING VALUE HANDLING
# =====================================================
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])

# =====================================================
# CLEANING
# =====================================================
df.drop(columns=["Loan_ID"], inplace=True)
df.dropna(subset=["Married"], inplace=True)

df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# =====================================================
# SPLIT FIRST (CRITICAL)
# =====================================================
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# ENCODE TARGET
# =====================================================
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)

# =====================================================
# LABEL ENCODING (FIT ON TRAIN ONLY)
# =====================================================
binary_cols = ["Gender", "Married", "Education", "Self_Employed"]

label_encoders = {}

for col in binary_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# =====================================================
# ONE-HOT ENCODING
# =====================================================
X_train = pd.get_dummies(X_train, columns=["Property_Area"], drop_first=True)
X_test = pd.get_dummies(X_test, columns=["Property_Area"], drop_first=True)

# Align columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

one_hot_cols = [c for c in X_train.columns if c.startswith("Property_Area_")]

# =====================================================
# LOG TRANSFORMATION (BEFORE SCALING)
# =====================================================
log_cols = ["ApplicantIncome", "CoapplicantIncome"]

for col in log_cols:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])

# =====================================================
# SCALING (FIT ON TRAIN ONLY)
# =====================================================
numerical_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# =====================================================
# SAVE PREPROCESSOR
# =====================================================
preprocessor = {
    "label_encoders": label_encoders,
    "target_encoder": target_encoder,
    "scaler": scaler,
    "numerical_cols": numerical_cols,
    "one_hot_cols": one_hot_cols,
    "feature_order": X_train.columns.tolist()
}

with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)

# =====================================================
# MODEL TRAINING
# =====================================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

# =====================================================
# SAVE MODEL
# =====================================================
with open(os.path.join(MODEL_DIR, "loan_approval_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

# =====================================================
# EVALUATION
# =====================================================
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print("Model training and saving completed.")
