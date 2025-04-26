from flask import Flask, request, render_template_string
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_preprocess_data
from concurrent.futures import ThreadPoolExecutor
import logging

app = Flask(__name__)

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Load models
with open('model/CatBoostRegressor_model.pkl', 'rb') as f:
    model_A = pickle.load(f)

with open('model/BaggingRegressor_3_model.pkl', 'rb') as f:
    model_B = pickle.load(f)

with open('model/StackingRegressor_model.pkl', 'rb') as f:
    model_C = pickle.load(f)
fallback_model = model_C

# Fit a scaler using training data
X_train, _, _, _ = load_and_preprocess_data()
scaler = StandardScaler().fit(X_train)

# Feature labels
feature_labels = [
    "Median Income",
    "House Age",
    "Average Rooms",
    "Average Bedrooms",
    "Population",
    "Average Occupancy",
    "Latitude",
    "Longitude"
]

# HTML form template
form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>California Housing Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        form { width: 400px; }
        input[type="text"] { width: 100%; padding: 8px; margin-bottom: 10px; }
        input[type="submit"] { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        h2 { color: #333; }
    </style>
</head>
<body>
  <h2>Enter Housing Feature Values</h2>
  <form method="POST" action="/predict">
    {% for i in range(8) %}
      <label>{{ feature_labels[i] }}</label><br>
      <input type="text" name="f{{i}}" required><br><br>
    {% endfor %}
    <input type="submit" value="Predict">
  </form>
</body>
</html>
"""

# Home route
@app.route('/')
def home():
    return render_template_string(form_html, feature_labels=feature_labels)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse user input
        features = np.array([[float(request.form[f"f{i}"]) for i in range(8)]])
        scaled_features = scaler.transform(features)

        # Predict using models in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(model.predict, scaled_features) for model in [model_A, model_B, model_C]]
            preds = [future.result()[0] for future in futures]

        # Ensemble prediction (mean)
        prediction = np.mean(preds)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        try:
            prediction = fallback_model.predict(scaler.transform(features))[0]
        except:
            return "<h2>Prediction Failed. Please check your input values.</h2>"

    return f"<h2>Predicted House Price (Aggregated): {prediction:.2f}</h2>"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
