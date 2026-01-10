# Loan Approval Prediction System

A machine learning–based web application that predicts whether a loan will be **approved or rejected** based on applicant details.  
The project uses a **Random Forest Classifier**, proper preprocessing, and a **Flask web interface** with a clean UI.

---

## Features

- End-to-end ML pipeline (training → inference)
- Proper handling of missing values
- Label encoding & one-hot encoding
- Log transformation & feature scaling
- Trained Random Forest model
- User-friendly Flask web application
- Dark-themed professional UI
- Loan term entered in **years** (converted internally to months)

---

## Machine Learning Details

- **Algorithm:** Random Forest Classifier
- **Target Variable:** `Loan_Status`
  - `Y` → Approved for Loan
  - `N` → Not Eligible for Loan
- **Key Preprocessing Steps:**
  - Missing value imputation (mean / mode)
  - Label Encoding for binary features
  - One-Hot Encoding for property area
  - Log transformation for income features
  - Standard scaling for numerical features
- **Class imbalance handled using:** `class_weight='balanced'`

---

## Project Structure
```
loan_approval/
│
├── data/
│ └── loan.csv
│
├── training/
│ └── model.py # Model training script
│
├── model/
│ ├── loan_approval_model.pkl
│ └── preprocessor.pkl
│
├── flask_app/
│ ├── app.py # Flask application
│ ├── templates/
│ │ └── index.html
│ └── static/
│ └── css/
│ └── styles.css
│
├── requirements.txt
└── README.md
```


---

## Web Application Inputs

- Gender
- Married
- Education
- Self Employed
- Dependents
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Term (Years)
- Credit History
- Property Area

---

## Model Performance

- **Training Accuracy:** ~100%
- **Test Accuracy:** ~83–85%

> High training accuracy is expected for Random Forest models.  
> Test accuracy indicates good generalization.

---

## How to Run the Project

### Clone the repository
```bash
git clone <your-repo-url>
cd loan_approval
```

### Create virtual environment (recommended)

python -m venv venv
venv\Scripts\activate   
Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies

pip install -r requirements.txt

# Train the model and run

cd training
python model.py

cd flask_app
python app.py




