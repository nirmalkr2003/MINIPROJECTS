# Car Price Prediction System

## Project Overview
This project predicts the selling price of a used car using machine learning based on features such as year, fuel type, transmission, and kilometers driven.

The project provides:
- A Flask web application
- A Streamlit interactive interface

---

## Project Structure
```
Flask project:
│   Dockerfile
│   Readme.md
│   requirements.txt
│   
├───data
│       car data.csv
│
├───flask_app
│   │   app.py
│   │
│   ├───static
│   │   ├───css
│   │   │       style.css
│   │   │
│   │   └───images
│   │           bg.jpg
│   │           image.jpeg
│   │
│   └───templates
│           index.html
│           result.html
│
├───model
│       Fuel_Type.pkl
│       model.pkl
│       scaling.pkl
│       Transmission.pkl
│
├───streamlit_app
│       main.py
│
├───training
│       model.py
│
├───UI_Screenshots
│       home_page1.png
│       home_page2.png
│       result_page.png
│
└───__pycache__
        streamlit.cpython-313.pyc 
```


## Run Flask Application

cd flask_app  
python app.py  


---

## Run Streamlit Application

cd streamlit_app  
streamlit run main.py

---

## Model Details
- Algorithm: Regression (scikit-learn)
- Model stored using pickle 

