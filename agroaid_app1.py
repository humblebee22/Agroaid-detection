import streamlit as st
import cv2
import numpy as np
import pickle
import json
from PIL import Image

# === Page Configuration === #
st.set_page_config(
    page_title="AgroAid - Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# === Custom CSS for Styling === #
st.markdown("""
<style>
    .title-style {
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        color: #2e7d32;
        margin-top: 0px;
        margin-bottom: 10px;
        padding-top: 1rem;
        position: relative;
        z-index: 999;
    }
    
    .subtitle-style {
        text-align: center;
        font-size: 20px;
        color: #4caf50;
        margin-bottom: 1em;
    }
    
    .intro-section {
        text-align: center;
        font-size: 25px;
        color: #FFFAFA;
        margin-bottom: 2em;
        padding: 0 2em;
    }
    
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border: none;
        padding: 0.5em 2em;
        border-radius: 12px;
        font-size: 18px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #66bb6a;
        transform: scale(1.05);
    }
    
    .upload-section {

        border-radius: 10px;
        font-family: "Poppins", sans-serif;
        padding: 0.5em;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-bottom: 0.5em;
        
    }
    
    .upload-section h3 {
        color: 	#FFFAFA !important;
        font-family: "Poppins", sans-serif !important;
        font-weight: 600 !important;
        font-size: 20px !important;
        
        margin-bottom: 0.5em !important;
    }
    
    .example-section {

        border-radius: 10px;
        font-family: "Poppins", sans-serif;
        padding: 0.5em;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-bottom: 0.5em;

    }
    hr {
    margin-top: -1em;
    margin-bottom: 1em;
}

    .example-section h3 {
        color: 	#FFFAFA !important;
        font-family: "Poppins", sans-serif !important;
        font-weight: 600 !important;
        font-size: 20px !important;
        
        margin-bottom: 0.5em !important;
    }
    
    .image-display {
        text-align: center;
        padding: 1em;
    }
    
    /* Remove default streamlit padding and ensure proper spacing */
    .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    .main .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# === Title === #
st.markdown('<div class="title-style">🌿 AgroAid</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Smart plant Disease Detector with Remedies</div>', unsafe_allow_html=True)

# === Introduction Section === #
st.markdown("""
<div class="intro-section">
    <p>Welcome to AgroAid - your intelligent companion for plant disease detection! Upload an image of a leaf or try our examples to get instant disease identification along with detailed remedies and prevention tips down below. Our advanced AI system helps farmers and gardeners maintain healthy crops.</p>
</div>
""", unsafe_allow_html=True)

# === Load Model and Metadata === #
try:
    model = pickle.load(open("D:/agroaidProject/models/final_model.pkl", "rb"))
    scaler = pickle.load(open("D:/agroaidProject/models/scaler.pkl", "rb"))
    pca = pickle.load(open("D:/agroaidProject/models/pca.pkl", "rb"))
    with open("D:/agroaidProject/disease_info.json", "r") as f:
        disease_data = json.load(f)
except Exception as e:
    st.error(f"⚠️ Error loading model files: {e}")
    st.stop()

# === Main Layout with Three Columns === #
col_left, col_center, col_right = st.columns([1, 1, 1])

# Initialize session state for storing results
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'display_image' not in st.session_state:
    st.session_state.display_image = None

# === Left Column: Upload Section === #
with col_left:
    st.markdown("""
    <div class="upload-section">
        <h3> UPLOAD LEAF IMAGE</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# === Right Column: Example Buttons Section === #
with col_right:
    st.markdown("""
    <div class="example-section">
        <h3> TRY EXAMPLES </h3>
    </div>
    """, unsafe_allow_html=True)
    
    example_options = {
        "Healthy Tomato": ("D:/agroaidProject/examples/tomato_healthy.jpg", "Tomato_healthy"),
        "Potato Late Blight": ("D:/agroaidProject/examples/potato_late_blight.jpg", "Potato___Late_blight"),
        "Pepper Bacterial Spot": ("D:/agroaidProject/examples/pepper_bacterial.jpg", "Pepper__bell___Bacterial_spot")
    }

    example_selected = None

    try:
        for label, (img_path, disease_key) in example_options.items():
            if st.button(label, key=f"example_{label}"):
                example_selected = (img_path, disease_key)
    except Exception as e:
        st.error(f"⚠️ Error setting up example buttons: {e}")

# === Center Column: Image Display === #
with col_center:
    st.markdown('<div class="image-display">', unsafe_allow_html=True)
    
    # === Prediction from Uploaded Image === #
    if uploaded_file:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_resized = cv2.resize(img, (100, 75))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            flat = gray.flatten().reshape(1, -1)
            scaled = scaler.transform(flat)
            reduced = pca.transform(scaled)
            prediction = model.predict(reduced)[0]

            disease_key = list(disease_data.keys())[prediction]
            info = disease_data[disease_key]

            # Store results in session state
            st.session_state.prediction_results = info
            st.session_state.display_image = img
            
            st.image(img, caption="Uploaded Leaf Image", width=300)
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
            st.session_state.prediction_results = None
            st.session_state.display_image = None

    # === Display Example Image === #
    elif example_selected is not None:
        try:
            img_path, disease_key = example_selected
            image = Image.open(img_path)
            info = disease_data[disease_key]
            
            # Store results in session state
            st.session_state.prediction_results = info
            st.session_state.display_image = image
            
            st.image(image, caption="Example Leaf Image", width=300)
            
        except Exception as e:
            st.error(f"⚠️ Error displaying example: {e}")
            st.session_state.prediction_results = None
            st.session_state.display_image = None
    
    else:
        st.info("👆 Upload an image or try an example to see results")
        st.session_state.prediction_results = None
        st.session_state.display_image = None
    
    st.markdown('</div>', unsafe_allow_html=True)

# === Results Section (Full Width) === #
if st.session_state.prediction_results is not None:
    st.markdown("---")
    
    try:
        info = st.session_state.prediction_results

        st.success(f"✅ Predicted Disease: **{info['disease']}**")
        st.info(f"🌾 Crop: **{info['crop']}**")

        st.markdown("#### 🧬 Cause")
        st.write(info["cause"])

        st.markdown("#### 💊 Remedies")
        for remedy in info["remedies"]:
            st.markdown(f"- {remedy}")

        st.markdown("#### 🛡️ Prevention")
        for tip in info["prevention"]:
            st.markdown(f"- {tip}")

        st.markdown("#### 🌱 Crop Care Advice")
        st.write(info["crop_care"])

    except Exception as e:
        st.error(f"⚠️ Error displaying results: {e}")