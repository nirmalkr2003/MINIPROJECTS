# import necessary libraries
# =====================================================
# IMPORTS
# =====================================================
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')


# =====================================================
# PATH SETUP (CRITICAL)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, 'car data.csv')


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)


# =====================================================
# OUTLIER HANDLING (IQR METHOD)
# =====================================================
outlier_cols = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven']

def remove_outliers_iqr(data, column):
    q1, q2, q3 = np.percentile(data[column], [25, 50, 75])
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    data[column] = np.where(
        data[column] > upper_limit, upper_limit,
        np.where(data[column] < lower_limit, lower_limit, data[column])
    )

for col in outlier_cols:
    remove_outliers_iqr(df, col)


# =====================================================
# FEATURE ENGINEERING
# =====================================================
df.drop(columns=['Car_Name'], inplace=True)

current_year = datetime.now().year

le_fuel = LabelEncoder()
le_trans = LabelEncoder()

df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

df = pd.get_dummies(df, columns=['Seller_Type'], drop_first=True)


# =====================================================
# SAVE ENCODERS
# =====================================================
with open(os.path.join(MODEL_DIR, 'Fuel_Type.pkl'), 'wb') as f:
    pickle.dump(le_fuel, f)

with open(os.path.join(MODEL_DIR, 'Transmission.pkl'), 'wb') as f:
    pickle.dump(le_trans, f)


# =====================================================
# SPLIT FEATURES & TARGET
# =====================================================
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


# =====================================================
# SCALING
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

with open(os.path.join(MODEL_DIR, 'scaling.pkl'), 'wb') as f:
    pickle.dump(scaler, f)


# =====================================================
# TRAIN–TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


# =====================================================
# MODEL TRAINING
# =====================================================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    min_samples_split=5
)

model.fit(X_train, y_train)


# =====================================================
# SAVE MODEL
# =====================================================
with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

print("Training completed successfully")
print("Model files saved in /model directory")
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2  = r2_score(y_test, y_pred)

# print(r2)
print("RUNNING MODEL.PY")
# =====================================================
