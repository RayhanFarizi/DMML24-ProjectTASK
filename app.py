from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the models and scaler
model_lr = joblib.load(open('linear_regression_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the request form
    features = [
        float(request.form['Number_of_Customers']),
        float(request.form['Menu_Price']),
        float(request.form['Marketing_Spend']),
        int(request.form['Cuisine_Type']),
        float(request.form['Average_Customer_Spending']),
        float(request.form['Promotions']),
        float(request.form['Reviews'])
    ]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict with both models
    prediction_lr = model_lr.predict(features_scaled)

    # Format predictions
    prediction = {
        'prediksi pendapatan anda': round(prediction_lr[0], 2),
    }

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)