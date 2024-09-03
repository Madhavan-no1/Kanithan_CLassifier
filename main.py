import streamlit as st
import os
import cv2
import numpy as np
import joblib
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import plotly.express as px
import random

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("GENAI_API_KEY")

# Configure GenAI API key
if api_key is None:
    st.error("API key is not set. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

# Load the trained waste classification model
model = joblib.load(r'C:\Users\AI_LAB\Downloads\Hack_hive_version2_using_gemini-main\new_svm_model.pkl')

# Define waste categories based on labels
recyclable_classes = {
    'cardboard',
    'glass', 'paper',
    'plastic_food_containers', 'plastic_soda_bottles',
    'steel_food_cans'
}

non_recyclable_classes = {
    'disposable_plastic_cutlery', 'plastic_shopping_bags', 'plastic_straws',
    'styrofoam_cups', 'styrofoam_food_containers', 'trash'
}

organic_classes = {
    'tea_bags'
}

other_classes = {
    'shoes'
}

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img.flatten()  # Flatten the image
    img = img.reshape(1, -1)  # Reshape for prediction
    return img

# Function to initialize the GenAI model
def initialize_model():
    generation_config = {"temperature": 0.5}  # Adjusted temperature for focused responses
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Function to generate waste-specific content with a focus on short, practical advice
def generate_short_disposal_content(model, waste_type):
    prompt = f"Provide practical guide for disposing of {waste_type} in India. Include bin color and common instruction of waste classification."
    response = model.generate_content([prompt])
    
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            text_part = candidate.content.parts[0]
            if text_part.text:
                return text_part.text
            else:
                return "No valid content generated."
        else:
            return "No content parts found."
    else:
        return "No candidates found."

# Placeholder function to simulate chat responses
def kanithan_chat_response(user_input):
    responses = {
        "hello": "Hello! How can I help you today?",
        "what can you do": "I can classify waste into recyclable, non-recyclable, and compostable categories!",
        "how to recycle": "You can recycle paper, plastics, and glass. Make sure to clean them before recycling!",
        "thank you": "You're welcome! Keep recycling!"
    }
    return responses.get(user_input.lower(), "I'm here to help! Please ask any questions you have about recycling.")

# Streamlit app setup
st.set_page_config(page_title="Kanithan - TheClassifier", layout="wide")

# Header and Title
st.markdown("<h1 style='text-align: center;'>Kanithan Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Classify your waste and help us recycle responsibly!</h3>", unsafe_allow_html=True)

# Display realistic animations for waste disposal
st.markdown("<div style='text-align: center;'><img src='https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzlzYzJzOHcybTQwaHQ0Z281aWJ2aWRvMXI2cnNyMnN5Ymt0bnQwYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2Z8gvu6xRbqCHA0bYh/giphy.webp' alt='Recycle Animation' width='200'></div>", unsafe_allow_html=True)

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an image of the item you want to classify:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and classify it
    processed_image = load_and_preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_name = prediction[0]  # Directly use the predicted label
    st.write(f"Predicted Class: **{predicted_class_name}**")

    # Determine the waste type and provide disposal instructions specific to India
    if predicted_class_name in recyclable_classes:
        waste_type = "recyclable waste"
        st.write("This item is **recyclable**. Place it in the **blue bin** for recycling.")
    elif predicted_class_name in non_recyclable_classes:
        waste_type = "non-recyclable waste"
        st.write("This item is **non-recyclable**. Dispose of it in the **red bin** for non-recyclable waste.")
    elif predicted_class_name in organic_classes:
        waste_type = "organic waste"
        st.write("This item is **organic**. Place it in the **green bin** for composting.")
    elif predicted_class_name in other_classes:
        waste_type = "special disposal waste"
        st.write("This item requires **special disposal**. Check local guidelines for proper disposal.")
    else:
        waste_type = "unknown type of waste"
        st.write("Recyclability status is **unknown**. Follow local guidelines for disposal.")

    # Initialize the GenAI model
    genai_model = initialize_model()

    # Generate short, practical waste-specific content
    content = generate_short_disposal_content(genai_model, waste_type)
    st.write("Suggested Disposal Method:")
    st.write(content)

    # User feedback
    feedback = st.radio("Was this classification correct?", ('Yes', 'No'))
    if feedback == 'No':
        correct_label = st.text_input("Please specify the correct category:")
        st.write(f"Thank you for your feedback! We'll use this to improve our classifier.")

# Kanithan Chat Box
st.markdown("## Kanithan Chat Box")
user_message = st.text_input("Type your question here:")
if user_message:
    response = kanithan_chat_response(user_message)
    st.write(f"Kanithan: {response}")

# Data Analytics Dashboard (simulated data)
st.markdown("## Waste Classification Analytics")
data = {
    'category': ['Recyclable', 'Non-Recyclable', 'Compostable'],
    'count': [150, 50, 30]  # Example data
}
fig = px.pie(data, values='count', names='category', title='Waste Classification Distribution')
st.plotly_chart(fig)

# Contact Us section
st.markdown("## Contact Us")
st.markdown("If you have any questions or need support, feel free to reach out to us:")
st.markdown("- Dhinesh: dhineshsaff@gmail.com")
st.markdown("- Hemachandaran: hemachandaran2520@gmail.com")
st.markdown("- Lavanya: lavanyagopal57@gmail.com")
st.markdown("- Madhavan: madhavanmdev@gmail.com")

# Sidebar content with more information
st.sidebar.title("Learn More")
st.sidebar.markdown("### Why Recycling is Important")
st.sidebar.markdown("- Recycling conserves natural resources and reduces pollution.")
st.sidebar.markdown("- It saves energy and reduces greenhouse gas emissions.")
st.sidebar.markdown("- Helps protect the environment for future generations.")

st.sidebar.markdown("### History of Recycling")
st.sidebar.markdown("The concept of recycling dates back to ancient times. People have been reusing materials for centuries. However, modern recycling began to take shape in the 1960s and 1970s when environmental concerns gained more attention.")
st.sidebar.markdown("Today, recycling is a critical part of sustainable living, reducing the waste that ends up in landfills and oceans.")

# Adding another realistic animation
st.sidebar.markdown("<div style='text-align: center;'><img src='https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGoybmZlbHoxM2E4cm9lZW1mb2syYmFscnZwcm5pY2c0b21tdXV2MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QaN5BogCfpp6hQ4GA9/giphy.webp' alt='Waste Disposal Animation' width='200'></div>", unsafe_allow_html=True)
