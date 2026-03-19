from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

with open("parkinsons_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        data["MDVP:Fo(Hz)"], data["MDVP:Fhi(Hz)"], data["MDVP:Flo(Hz)"],
        data["MDVP:Jitter(%)"], data["MDVP:Jitter(Abs)"], data["MDVP:RAP"],
        data["MDVP:PPQ"], data["Jitter:DDP"], data["MDVP:Shimmer"],
        data["MDVP:Shimmer(dB)"], data["Shimmer:APQ3"], data["Shimmer:APQ5"],
        data["MDVP:APQ"], data["Shimmer:DDA"], data["NHR"], data["HNR"],
        data["RPDE"], data["DFA"], data["spread1"], data["spread2"],
        data["D2"], data["PPE"]
    ]
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(probability) * 100, 2),
        "result": "Parkinson's Detected" if prediction == 1 else "Healthy"
    })

@app.route("/", methods=["GET"])
def home():
    return "NeuroVoice Backend is Running! ✅"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
