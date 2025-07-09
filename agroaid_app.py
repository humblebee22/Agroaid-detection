import streamlit as st
from PIL import Image
import numpy as np
import os
import pickle
import json
import cv2

# Load disease info
with open("disease_info.json", "r") as f:
    disease_data = json.load(f)

# Load model and preprocessing
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("models/final_model.pkl", "rb") as f:
    model = pickle.load(f)

# App UI
st.set_page_config(page_title="AgroAid - Plant Disease Detector")
st.title("🌿 AgroAid: Smart Plant Disease Detection")

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Convert PIL to OpenCV format
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (100, 100))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    flat = img_gray.flatten().reshape(1, -1)

    # Preprocess and predict
    scaled = scaler.transform(flat)
    reduced = pca.transform(scaled)
    prediction = model.predict(reduced)[0]

    # Get label and details
    predicted_label = list(disease_data.keys())[prediction]
    info = disease_data[predicted_label]

    # Display result
    st.subheader(f"🌱 Predicted Crop: {info['crop']}")
    st.write(f"🦠 **Disease**: {info['disease']}")
    st.write(f"⚠️ **Cause**: {info['cause']}")
    if info["remedies"]:
        st.write("💊 **Remedies:**")
        for remedy in info["remedies"]:
            st.write(f" - {remedy}")
    if info["prevention"]:
        st.write("🛡️ **Prevention:**")
        for prev in info["prevention"]:
            st.write(f" - {prev}")
    st.write(f"🧑‍🌾 **Crop Care Tip:** {info['crop_care']}")
