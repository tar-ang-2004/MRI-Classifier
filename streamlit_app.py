import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from gtts import gTTS
from playsound import playsound
import tempfile
import os
import time
import random
import requests

DEVICE = torch.device('cpu')
class_names = ["glioma", "meningioma", "no tumor", "pituitary"]

MODEL_PATH = "models/transfer_model.pt"
GDRIVE_FILE_ID = "1kb6mE0dgcftsHxZjgQxhFZF3QjzvpJgs"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        response = requests.get(url, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)

@st.cache_resource
def load_model():
    download_model()
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        flipped_tensor = transforms.functional.hflip(input_tensor)
        outputs_flipped = model(flipped_tensor)
        outputs_avg = (outputs + outputs_flipped) / 2
        _, pred = torch.max(outputs_avg, 1)
        conf = torch.softmax(outputs_avg, 1)[0][pred].item() * 100
    return class_names[pred], conf

def speak(text):
    tts = gTTS(text, slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        playsound(fp.name)
    os.remove(fp.name)

# --- CSS UI ---
st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        background: linear-gradient(to right, #ff6ec4, #7873f5);
        -webkit-background-clip: text;
        color: transparent;
        animation: float 3s ease-in-out infinite;
        text-align: center;
        margin-bottom: 20px;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    .warning {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white; padding: 1em;
        border-radius: 1em; margin-bottom: 1em;
        animation: float 1.5s ease-in-out infinite;
    }
    .result {
        background: linear-gradient(to right, #96e6a1, #d4fc79);
        color: black; font-weight: bold; font-size: 1.1em;
        padding: 1em; border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .fadeout-box {
        animation: fadeout 1.2s ease-out forwards;
        font-size: 1.1em;
        background: linear-gradient(to right, #f2709c, #ff9472);
        padding: 1em; border-radius: 12px;
        text-align: center; color: white;
        margin-top: 20px; font-weight: bold;
    }
    @keyframes fadeout {
        0% {opacity: 1;} 50% {opacity: 0.5;} 100% {opacity: 0;}
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">🧠 Brain Tumor MRI Classifier</div>', unsafe_allow_html=True)

# --- Session State ---
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = 'Single'
if 'speak_enabled' not in st.session_state:
    st.session_state.speak_enabled = False
if 'fade_reset' not in st.session_state:
    st.session_state.fade_reset = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(random.randint(1000, 9999))

# --- Button Controls ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📷 Single Image"):
        st.session_state.upload_mode = "Single"
with col2:
    if st.button("🖼️ Multiple Images"):
        st.session_state.upload_mode = "Multiple"
with col3:
    if st.button("🔊 Toggle Speak"):
        st.session_state.speak_enabled = not st.session_state.speak_enabled
with col4:
    if st.button("🧹 Clear / Reset"):
        st.session_state.fade_reset = True
        st.session_state.upload_mode = "Single"
        st.session_state.speak_enabled = False
        st.session_state.uploader_key = str(random.randint(1000, 9999))

# --- Reset Animation ---
if st.session_state.fade_reset:
    st.markdown('<div class="fadeout-box">🔄 Resetting... Please wait...</div>', unsafe_allow_html=True)
    time.sleep(1.2)
    st.session_state.fade_reset = False
    st.rerun()

# --- Upload Info ---
st.markdown(f"**Upload Mode:** `{st.session_state.upload_mode}`")
st.markdown(f"**Speak Enabled:** `{st.session_state.speak_enabled}`")

# --- Uploader ---
files = st.file_uploader("Upload MRI image(s)", type=['jpg', 'jpeg', 'png'],
                         accept_multiple_files=(st.session_state.upload_mode == "Multiple"),
                         key=st.session_state.uploader_key)

# --- Predict Loop ---
if files:
    if st.session_state.upload_mode == "Single":
        files = [files]
    for idx, file in enumerate(files):
        image = Image.open(file)
        if image.mode != 'RGB':
            st.markdown(f'<div class="warning">⚠️ Image {idx+1} is not RGB. Please upload RGB images only.</div>', unsafe_allow_html=True)
            if st.session_state.speak_enabled:
                speak(f"Image {idx+1} is not RGB.")
            continue
        st.image(image, caption=f"🖼️ MRI Image {idx+1}", use_column_width=True)
        pred, conf = predict_image(image)
        result_text = f"Prediction: {pred.upper()} | Confidence: {conf:.2f}%"
        st.markdown(f'<div class="result">{result_text}</div>', unsafe_allow_html=True)
        if st.session_state.speak_enabled:
            speak(f'Prediction for image {idx+1}: {pred.upper()}')
        st.markdown("---")
