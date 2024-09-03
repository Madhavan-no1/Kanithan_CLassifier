import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_images_from_folder(folder):
    images = []
    labels = []
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if not os.path.isdir(category_path):
            continue
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            img = cv2.resize(img, (128, 128))
            img = img.flatten()
            images.append(img)
            labels.append(category)
    return np.array(images), np.array(labels)

# Load the new model
model = joblib.load('new_svm_model.pkl')

# Load test data
test_images, test_labels = load_images_from_folder(r"C:\Users\AI_LAB\Downloads\Hack_hive_version2_using_gemini-main\dataset")

# Make predictions
predictions = model.predict(test_images)

# Evaluate the new model
print(classification_report(test_labels, predictions))
print(f"Accuracy: {accuracy_score(test_labels, predictions)}")
