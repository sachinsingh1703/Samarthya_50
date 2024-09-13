import json
import numpy as np
import keras
import requests
from io import BytesIO
from PIL import Image

def load_keras_model():
    with open("lib/hyperparameters.json") as file:
        hyperparameters = json.load(file)

    model = keras.models.load_model('HTF.h5')
    return model

def fetch_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def make_predictions(url):
    model = load_keras_model()
    image = fetch_image_from_url(url)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    _, predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]