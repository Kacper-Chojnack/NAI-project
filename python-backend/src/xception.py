import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.xception import Xception
from sklearn.model_selection import train_test_split

# Wczytanie obrazków i ich etykiet
def load_images_and_labels(img_dir, labels_file, max_images_per_class=60):
    images = []
    labels = []

    # Wczytanie nazw klas
    with open(labels_file, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    # Dla każdej klasy wczytujemy obrazy
    for class_name in class_names:
        class_path = os.path.join(img_dir, class_name)
        if os.path.isdir(class_path):
            class_index = class_to_index[class_name]

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
                        img = tf.keras.utils.img_to_array(img)
                        images.append(img)
                        labels.append(class_index)
                    except Exception as e:
                        print(f"Problem z obrazkiem {img_name}: {e}")

    return np.array(images), np.array(labels)

# Ścieżki do danych
img_dir = '../sample-image-data/animals/animals/'
labels_file = '../sample-image-data/name_of_the_animals.txt'

# Ładowanie danych
images, labels = load_images_and_labels(img_dir, labels_file)

# Dzielimy dane na treningowe i walidacyjne
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Normalizacja (przeskalowanie pikseli do zakresu 0-1)
train_images = train_images / 255.0
val_images = val_images / 255.0

# Tworzenie modelu Xception
def create_xception_model(num_classes):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False  # Zamrażanie początkowych warstw

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Inicjalizacja modelu
xception_model = create_xception_model(len(set(labels)))

# Trening modelu
xception_history = xception_model.fit(
    train_images, train_labels,
    batch_size=32,
    epochs=10,  # Krótki trening na testy
    validation_data=(val_images, val_labels),
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

# Zapisujemy model
os.makedirs('../models', exist_ok=True)
xception_model.save('../models/xception_model.keras')

print("Model zapisany w ../models/xception_model.keras")
