from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

app = Flask(__name__)

model = load_model(r"D:\fyp\GEN\gender 1.0\gender-detection-keras-master\gender-detection-keras-master\pre-trained\gender_detection.model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (96, 96))
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]

    gender = "Man" if prediction[0] > prediction[1] else "Woman"

    return jsonify({'gender': gender})

if __name__ == '__main__':
    app.run(debug=True)
