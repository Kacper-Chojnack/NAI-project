from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Tworzymy aplikację Flask
app = Flask(__name__)

# Wczytujemy wcześniej wytrenowane modele
mobilenet_model = tf.keras.models.load_model("../models/mobilenet_model.keras")
nasnetmobile_model = tf.keras.models.load_model("../models/nasnetmobile_model.keras")
xception_model = tf.keras.models.load_model("../models/xception_model.keras")

# Lista klas (zwierzęta, które modele rozpoznają)
class_names = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat", "caterpillar",
    "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly",
    "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper",
    "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo",
    "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus", "okapi",
    "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", "pig", "pigeon",
    "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark",
    "sheep", "snake", "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale",
    "wolf", "wombat", "woodpecker", "zebra"
]

# Funkcja do predykcji obrazu za pomocą modelu
def predict_with_model(model, image, class_names):
    image = np.expand_dims(image, axis=0)  # Dodajemy wymiar batch (potrzebny dla modelu)
    predictions = model.predict(image)  # Model robi przewidywanie
    max_prob = np.max(predictions[0])  # Największe prawdopodobieństwo
    predicted_class = class_names[np.argmax(predictions[0])]  # Klasa z największym prawdopodobieństwem
    return predicted_class, max_prob

# Endpoint do przewidywania
@app.route("/predict", methods=["POST"])
def predict():
    # Sprawdzamy, czy przesłano plik
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Wczytujemy obraz z przesłanego pliku
        image = Image.open(io.BytesIO(file.read()))

        # Jeśli obraz nie jest w formacie RGB, konwertujemy go
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Skalujemy obraz do wymiarów 256x256
        image = image.resize((256, 256))

        # Przekształcamy obraz na macierz NumPy i normalizujemy wartości pikseli (0-1)
        image_array = np.array(image).astype(np.float32) / 255.0

        # Robimy przewidywania dla każdego modelu
        mobilenet_result, mobilenet_prob = predict_with_model(mobilenet_model, image_array, class_names)
        nasnetmobile_result, nasnetmobile_prob = predict_with_model(nasnetmobile_model, image_array, class_names)
        xception_result, xception_prob = predict_with_model(xception_model, image_array, class_names)

        # Przygotowujemy wyniki
        results = {
            "MobileNetV2": {"prediction": mobilenet_result, "probability": round(float(mobilenet_prob), 2)},
            "NASNetMobile": {"prediction": nasnetmobile_result, "probability": round(float(nasnetmobile_prob), 2)},
            "Xception": {"prediction": xception_result, "probability": round(float(xception_prob), 2)}
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Uruchamiamy serwer
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
