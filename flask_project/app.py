from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained files
model = pickle.load(open('model.pkl', 'rb'))
Fuel_Type_en = pickle.load(open('Fuel_Type.pkl', 'rb'))
Transmission_en = pickle.load(open('Transmission.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

print("Model and encoders loaded")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = float(request.form['Kms_Driven'])
        Fuel_Type = int(request.form['Fuel_Type'])
        Transmission = request.form['Transmission']
        Owner = int(request.form['Owner'])
        Seller_Type_Individual = int(request.form['Seller_Type_Individual'])

        # Encode Transmission
        Transmission_val = Transmission_en.transform([Transmission])[0]

        # Prepare input
        details = [
            Year,
            Present_Price,
            Kms_Driven,
            Fuel_Type,
            int(Transmission_val),
            Owner,
            Seller_Type_Individual
        ]

        data_out = np.array(details).reshape(1, -1)
        data_scaled = scaler.transform(data_out)

        # Predict
        prediction = model.predict(data_scaled)[0]

        return render_template(
            'result.html',
            price=round(prediction, 2)
        )

    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
