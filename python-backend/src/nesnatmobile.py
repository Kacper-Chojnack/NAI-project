import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import NASNetMobile
from sklearn.model_selection import train_test_split

# Funkcja do ładowania obrazów i etykiet
def load_images_and_labels(img_dir, labels_file, max_images_per_class=60):
    images = []
    labels = []

    with open(labels_file, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

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
                        print(f"Warning: Cannot process image {img_name}: {e}")

    return np.array(images), np.array(labels)

# Ścieżki do danych
img_dir = '../sample-image-data/animals/animals/'
labels_file = '../sample-image-data/name_of_the_animals.txt'

# Ładowanie danych
images, labels = load_images_and_labels(img_dir, labels_file)

# Podział danych
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Normalizacja danych
train_images = train_images / 255.0
val_images = val_images / 255.0

# Definicja modelu
def create_nasnetmobile_model(num_classes):
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Inicjalizacja modelu
nasnetmobile_model = create_nasnetmobile_model(len(set(labels)))

# Trenowanie modelu
nasnetmobile_history = nasnetmobile_model.fit(
    train_images, train_labels,
    batch_size=16,  # Zmniejszenie batch size dla małego zbioru danych
    epochs=20,  # Więcej epok dla małego zbioru danych
    validation_data=(val_images, val_labels),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Zapisywanie modelu
os.makedirs('../models', exist_ok=True)
nasnetmobile_model.save('../models/nasnetmobile_model.keras')
