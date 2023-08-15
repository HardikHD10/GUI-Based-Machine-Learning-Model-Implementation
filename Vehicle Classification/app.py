import os
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io


app = Flask(__name__)

model = load_model('vehicle.h5')
categories = ['vehicles', 'non-vehicles']

def preprocess_image(image):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    img_expanded = np.expand_dims(img_resized, axis=0)
    return img_expanded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']
    image = Image.open(file.stream)
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_category = categories[int(np.round(prediction[0][0]))]

    return f"This image is likely to contain a: {predicted_category}"

if __name__ == '__main__':
    app.run(debug=True)
