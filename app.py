from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained model
model = joblib.load('decision_tree_drugs_predict.pkl') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = int(data.get('age'))
        na_to_k = float(data.get('na_to_k'))
        sex = data.get('sex')
        bp = data.get('bp')
        cholesterol = data.get('cholesterol')

        # Validate numerical inputs
        if age is None or na_to_k is None:
            return jsonify({"error": "Missing required numerical inputs: age or Na_to_K"}), 400

        if not isinstance(age, (int, float)) or not isinstance(na_to_k, (int, float)):
            return jsonify({"error": "Invalid input types for age or Na_to_K. They must be numbers."}), 400

        # Validate categorical inputs
        if sex not in ["Male", "Female"]:
            return jsonify({"error": "Invalid value for Sex. Must be 'Male' or 'Female'."}), 400
        if bp not in ["Low", "Normal", "High"]:
            return jsonify({"error": "Invalid value for BP. Must be 'Low', 'Normal', or 'High'."}), 400
        if cholesterol not in ["Normal", "High"]:
            return jsonify({"error": "Invalid value for Cholesterol. Must be 'Normal' or 'High'."}), 400

        # Encode categorical values
        sex_encoded = 1 if sex == "Male" else 0
        bp_encoded = {"Low": 0, "Normal": 1, "High": 2}[bp]
        cholesterol_encoded = {"Normal": 0, "High": 1}[cholesterol]

        # Prepare input data for the model
        input_data = np.array([[age, na_to_k, sex_encoded, bp_encoded, cholesterol_encoded]])

        # Predict using the pre-trained model
        prediction = model.predict(input_data)[0]

        # Convert prediction to native Python int if it's a NumPy type
        if isinstance(prediction, (np.integer, np.int64, np.int32)):
            prediction = int(prediction)
        elif isinstance(prediction, (np.floating, np.float64, np.float32)):
            prediction = float(prediction)

        predictionName = ["DrugA", "DrugB", "DrugC", "DrugX", "DrugY"]

        return jsonify({"prediction": predictionName[prediction]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

