import streamlit as st
import requests
import base64
import json
from PIL import Image
import io

# Flask API URL
FLASK_API_URL = "http://localhost:8080/predict"  # Change this if running Flask on a different host

st.title("Image Classification App")
st.write("Upload an image and get predictions!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Send request to Flask API
    if st.button("Get Prediction"):
        response = requests.post(FLASK_API_URL, json={"image": encoded_image})
        
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction: " + str(result))
        else:
            st.error("Error in prediction. Check Flask server.")
