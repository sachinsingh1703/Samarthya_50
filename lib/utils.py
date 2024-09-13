import json
import numpy as np
import keras

def load_keras_model():
    with open("lib/hyperparameters.json") as file:
        hyperparameters = json.load(file)

    model = keras.models.load_model('HTF.h5')
    return model

def make_predictions(image_file):
    model = load_keras_model()
    image = Image.open(image_file).convert('RGB')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    _, predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]