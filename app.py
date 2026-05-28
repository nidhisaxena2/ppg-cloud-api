from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/model.keras")

@app.route("/")
def home():
    return "PPG API Running Successfully"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["signal"]
        data = np.array(data).reshape(1, 4000, 1)

        pred = model.predict(data)[0][0]

        return jsonify({
            "prediction": float(pred)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# IMPORTANT: Render port fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
