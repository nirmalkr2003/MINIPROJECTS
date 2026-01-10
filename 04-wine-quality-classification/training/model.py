import os
import pickle
import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "wine.csv")

df = pd.read_csv(CSV_PATH)

X = df.drop("type", axis=1)
y = df["type"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)



rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_model.fit(X_train, y_train)

# =====================================================
# SAVE MODEL & ENCODER
# =====================================================
with open(os.path.join(MODEL_DIR, "wine_type_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

with open(os.path.join(MODEL_DIR, "wine_type_encoder.pkl"), "wb") as f:
    pickle.dump(target_encoder, f)

print("Wine type model training and saving completed.")
print("Train accuracy:", rf_model.score(X_train, y_train))
print("Test accuracy :", rf_model.score(X_test, y_test))

