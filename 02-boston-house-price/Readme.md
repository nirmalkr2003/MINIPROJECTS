# Boston Housing Price Prediction

This project predicts the **median house value (MEDV)** in Boston neighborhoods using a **Gradient Boosting Regressor**.  
The model is trained using engineered features and deployed through a **Streamlit web application** with a user-friendly interface.

---

## Project Overview

- Dataset: Boston Housing Dataset
- Algorithm: Gradient Boosting Regressor
- Task: Regression (House price prediction)
- Deployment: Streamlit
- Language: Python

The project follows a **clean ML workflow**:
1. Data preprocessing & feature engineering
2. Model training using best hyperparameters
3. Saving trained model & scaler
4. Interactive prediction using Streamlit UI

---

## Project Structure
```
project_root/
│
├── data/
│ └── BostonHousing.csv
│
├── model/
│ ├── gbr_model.pkl
│ └── scaling.pkl
│
├── training/
│ └── model.py
│
├── streamlit_app/
│ └── boston_app.py
│
├── requirements.txt
└── README.md
```


---

## Feature Engineering

The following transformations are applied:

- Dropped column: `indus`
- Removed duplicate rows
- Created new features:
  - `rooms_per_age = rm / (age + 1)`
  - `tax_per_room = tax / rm`
- Standard scaling using `StandardScaler`

These steps are **replicated exactly** during prediction to ensure consistency.

---

##  Model Details

- Algorithm: Gradient Boosting Regressor
- Best Hyperparameters (from tuning):
  ```python
  learning_rate = 0.1
  max_depth = 5
  n_estimators = 100
  subsample = 0.8

## Install dependencies
pip install -r requirements.txt

## Train the model
cd training
python model.py


This will:
Train the Gradient Boosting model

Save gbr_model.pkl and scaling.pkl into the model/ folder

## Run the Streamlit app
cd streamlit_app
streamlit run boston_app.py

Open the provided local URL in your browser.