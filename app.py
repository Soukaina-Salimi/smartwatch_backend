from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Firebase (optionnel)
import firebase_admin
from firebase_admin import credentials, db

cred_path = "firebase-adminsdk.json"  # remplace si tu veux utiliser Firebase
if os.path.exists(cred_path):
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smartwatch-iot-1a52c-default-rtdb.firebaseio.com/'  # remplace par ton URL
    })

app = Flask(__name__)

# Charger le modèle Random Forest
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [
            data["heart_rate"],
            data["temperature"],
            data["uv"],
            data["movement"]
        ]
        prediction = model.predict([features])[0]

        # Optionnel : sauvegarder dans Firebase
        if os.path.exists(cred_path):
            ref = db.reference("/predictions")
            ref.push({
                "heart_rate": data["heart_rate"],
                "temperature": data["temperature"],
                "uv": data["uv"],
                "movement": data["movement"],
                "prediction": int(prediction)
            })

        return jsonify({"prediction": int(prediction), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)