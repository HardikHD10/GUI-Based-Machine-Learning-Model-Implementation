from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

app = Flask(__name__)

# Load the image captioning model
model_path = "/Users/hardik/Documents/Semester6/AI Lab/image captioning/model.h5"
model = load_model(model_path)

# Load the tokenizer
# Replace with your actual tokenizer or load it from a file
tokenizer = Tokenizer()

max_length = 50  # Update this to the maximum length of your captions
img_size = 224

def preprocess_image(image_path, img_size=224):
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def generate_caption(model, tokenizer, image, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([image, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = tokenizer.index_word[y_pred]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            os.makedirs('static', exist_ok=True)
            image_path = os.path.join('static', file.filename)
            file.save(image_path)
            image = preprocess_image(image_path, img_size)
            caption = generate_caption(model, tokenizer, image, max_length)
            result = {
                'caption': caption,
                'image_path': image_path
            }
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
