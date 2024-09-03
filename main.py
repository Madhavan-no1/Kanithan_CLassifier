import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the custom VGG-16 model for waste segregation
model_path = r"C:\Users\AI_LAB\Downloads\Recyclewaste2-maintensorflow\vgg16_waste_segregate.keras"
model = load_model(model_path)

st.title("Waste Segregation using VGG-16")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)

    # You may need to adjust the labels based on your model's training
    labels = ["Class 1: Label", "Class 2: Label", "Class 3: Label"]  # Replace with your actual labels

    st.write(f"Predicted Class: {labels[predicted_class[0]]}")
