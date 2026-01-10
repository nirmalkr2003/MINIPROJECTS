import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, 'BostonHousing.csv')
# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)

df.drop(columns=['indus'], inplace=True)
df = df.drop_duplicates()
df['rooms_per_age'] = df['rm'] / (df['age'] + 1)
df['tax_per_room'] = df['tax'] / df['rm']

x = df.drop('medv', axis=1)
y = df['medv']

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

x = pd.DataFrame(X_scaled, columns=x.columns)

with open(os.path.join(MODEL_DIR, 'scaling.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# TRAIN-TEST SPLIT

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# from sklearn.ensemble import RandomForestRegressor

# rfr = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=5,
#     random_state=42)

# rfr.fit(x_train, y_train)
# y_pred = rfr.predict(x_test)


# =====================================================
# MODEL TRAINING
# =====================================================


from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score

gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    subsample=0.8,
)
gbr.fit(x_train, y_train)
# y_pred = gbr.predict(x_test)


# mse = mean_squared_error(y_test, y_pred)


# =====================================================
# SAVE MODEL
# =====================================================

with open(os.path.join(MODEL_DIR, 'gbr_model.pkl'), 'wb') as f:
    pickle.dump(gbr, f)
print("Training completed successfully")
print("Model files saved in /model directory")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = gbr.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(r2)
print("Mean Absolute Error:", mae)
