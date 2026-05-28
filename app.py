
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("model/model.keras")

@app.route("/")
def home():
    return "PPG API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["signal"]
    data = np.array(data).reshape(1, 4000, 1)

    pred = model.predict(data)[0][0]

    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
