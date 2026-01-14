import numpy as np
import random
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

LOOK_BACK = 100

# Load model from project directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stockPrice.keras")

app = Flask(__name__)
CORS(app)   # ✅ REQUIRED for Netlify → Render

# Load model once at startup
model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_sequence = data.get("sequence", [])
        min_limit = float(data.get("min_limit"))
        max_limit = float(data.get("max_limit"))

        if min_limit >= max_limit:
            return jsonify({"error": "Min limit must be less than Max limit"}), 400

        # Auto-generate sequence if not provided
        if not input_sequence:
            input_sequence = [
                random.uniform(min_limit, max_limit)
                for _ in range(LOOK_BACK)
            ]

        if len(input_sequence) != LOOK_BACK:
            return jsonify({
                "error": f"Exactly {LOOK_BACK} values required"
            }), 400

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array([[min_limit], [max_limit]]))

        arr = np.array(input_sequence).reshape(-1, 1)
        scaled = scaler.transform(arr)
        reshaped = scaled.reshape(1, LOOK_BACK, 1)

        prediction = model.predict(reshaped, verbose=0)
        final_price = scaler.inverse_transform(
            prediction.reshape(-1, 1)
        )[0][0]

        return jsonify({
            "status": "success",
            "predicted_price": float(final_price)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
