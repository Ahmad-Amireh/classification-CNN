from flask import Flask, request, jsonify,render_template
import os
from flask_cors import CORS
from CNNproject.utils.common import decodeImage
from CNNproject.pipeline.predict import PredictionPipeline

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()  # Instantiate before defining routes

@app.route("/", methods=["GET"])
def home():
    #return jsonify({"message": "Flask API is running!"})
    return render_template("index.html")

@app.route("/train", methods=["GET", "POST"])
def train_Route():
    os.system("dvc repro")  # Consider using subprocess.run()
    return jsonify({"message": "Training Done Successfully!"})

@app.route("/predict", methods=["POST"])
def predictRoute():
    try:
        image = request.json["image"]
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
