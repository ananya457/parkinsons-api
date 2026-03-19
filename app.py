from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Create the Flask app
app = Flask(__name__)
CORS(app)  # This allows your HTML frontend to talk to this backend

# -----------------------------------------------
# Load your trained model and scaler when server starts
# -----------------------------------------------
with open("parkinsons_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------------------------
# This is the prediction route
# Your frontend will send data here
# -----------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Get the JSON data sent from the frontend
    data = request.get_json()

    # Put all 22 voice features into a list (in the correct order)
    features = [
        data["MDVP:Fo(Hz)"],
        data["MDVP:Fhi(Hz)"],
        data["MDVP:Flo(Hz)"],
        data["MDVP:Jitter(%)"],
        data["MDVP:Jitter(Abs)"],
        data["MDVP:RAP"],
        data["MDVP:PPQ"],
        data["Jitter:DDP"],
        data["MDVP:Shimmer"],
        data["MDVP:Shimmer(dB)"],
        data["Shimmer:APQ3"],
        data["Shimmer:APQ5"],
        data["MDVP:APQ"],
        data["Shimmer:DDA"],
        data["NHR"],
        data["HNR"],
        data["RPDE"],
        data["DFA"],
        data["spread1"],
        data["spread2"],
        data["D2"],
        data["PPE"]
    ]

    # Convert the list into a numpy array (format the model understands)
    features_array = np.array(features).reshape(1, -1)

    # Scale the features using your scaler
    scaled_features = scaler.transform(features_array)

    # Run the model prediction
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    # Send the result back to the frontend
    return jsonify({
        "prediction": int(prediction),                        # 1 = Parkinson's, 0 = Healthy
        "probability": round(float(probability) * 100, 2),   # e.g. 87.5 (percent)
        "result": "Parkinson's Detected" if prediction == 1 else "Healthy"
    })

# -----------------------------------------------
# A simple test route — open in browser to check if server is running
# -----------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "NeuroVoice Backend is Running! ✅"

# -----------------------------------------------
# Start the server
# -----------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
