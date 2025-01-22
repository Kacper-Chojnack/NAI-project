import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Wczytujemy obrazki i przypisujemy etykiety
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
                        print(f"Problem z obrazkiem {img_name}: {e}")

    return np.array(images), np.array(labels)

# Testowanie modelu na danych
def evaluate_model(model_path, test_images, test_labels, labels_file):
    print(f"\nModel: {model_path}")
    model = tf.keras.models.load_model(model_path)  # Wczytujemy model
    predictions = model.predict(test_images)  # Model przewiduje wyniki
    predicted_labels = np.argmax(predictions, axis=1)  # Wybieramy klasy z najwyższym wynikiem

    # Liczymy metryki
    precision = precision_score(test_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predicted_labels, average='weighted', zero_division=0)

    # Wyświetlamy dokładne wyniki
    print("\nSzczegółowe wyniki:")
    class_names = open(labels_file).read().splitlines()
    available_classes = sorted(set(test_labels))  # Tylko klasy obecne w danych testowych
    available_class_names = [class_names[i] for i in available_classes]

    print(classification_report(
        test_labels,
        predicted_labels,
        labels=available_classes,
        target_names=available_class_names,
        zero_division=0
    ))

    return precision, recall, f1

# Dane testowe
test_img_dir = '../sample-image-data/animals-test-data/animals/'
labels_file = '../sample-image-data/name_of_the_animals.txt'

# Ładowanie obrazków testowych i etykiet
test_images, test_labels = load_images_and_labels(test_img_dir, labels_file)

# Normalizacja obrazków (tak jak przy trenowaniu)
test_images = test_images / 255.0

# Lista modeli do sprawdzenia
model_paths = [
    '../models/nasnetmobile_model.keras',
    '../models/mobilenet_model.keras',
    '../models/xception_model.keras'
]

# Porównanie wyników dla każdego modelu
results = []

for model_path in model_paths:
    precision, recall, f1 = evaluate_model(model_path, test_images, test_labels, labels_file)
    results.append({
        'Model': os.path.basename(model_path),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Podsumowanie wyników
print("\nPorównanie modeli:")
for result in results:
    print(f"Model: {result['Model']}")
    print(f"Precyzja: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1 Score: {result['F1 Score']:.4f}")
    print()

# Tworzenie wykresu porównawczego
models = [result['Model'] for result in results]
precision_scores = [result['Precision'] for result in results]
recall_scores = [result['Recall'] for result in results]
f1_scores = [result['F1 Score'] for result in results]

x = np.arange(len(models))  # Lokalizacja dla grup na osi X

plt.figure(figsize=(10, 6))
bar_width = 0.2

plt.bar(x - bar_width, precision_scores, bar_width, label='Precision')
plt.bar(x, recall_scores, bar_width, label='Recall')
plt.bar(x + bar_width, f1_scores, bar_width, label='F1 Score')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Porównanie wyników modeli')
plt.xticks(x, models)
plt.legend()

plt.tight_layout()
plt.show()
