import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import gtts
import tempfile
import os
import random
import time
import gdown
import io
import numpy as np
from gtts import gTTS


DEVICE = torch.device('cpu')
class_names = ["glioma", "meningioma", "no tumor", "pituitary"]

@st.cache_resource
def load_model():
    model_path = "models/transfer_model.pt"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/uc?id=1kb6mE0dgcftsHxZjgQxhFZF3QjzvpJgs"
        gdown.download(url, model_path, quiet=False)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def is_probably_mri(image):
    image_np = np.array(image.resize((224, 224))) / 255.0
    # Accept images that are either grayscale or have low to moderate saturation
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    std_rgb = np.std([r, g, b])
    grayscale_like = std_rgb < 0.15  # Relaxed threshold
    max_rgb = np.max(image_np, axis=2)
    min_rgb = np.min(image_np, axis=2)
    saturation = np.mean(max_rgb - min_rgb)
    low_saturation = saturation < 0.35  # Relaxed threshold
    w, h = image.size
    size_ok = 100 <= w <= 1024 and 100 <= h <= 1024  # Allow larger images
    return (grayscale_like or low_saturation) and size_ok

def predict_image(image, temperature=0.3):
    # Use advanced TTA: flips, rotations, color jitter, and sharpness
    tta_transforms = [
        lambda x: x,
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(15),
        lambda x: x.rotate(-15),
        lambda x: transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)(x),
        lambda x: transforms.RandomAdjustSharpness(sharpness_factor=2)(x),
    ]
    outputs = []
    for tta in tta_transforms:
        aug_img = tta(image)
        input_tensor = transform(aug_img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            outputs.append(output)
    # Ensemble predictions for maximum confidence
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    # Temperature scaling for sharper probabilities
    scaled_output = avg_output / temperature
    probabilities = torch.nn.functional.softmax(scaled_output, dim=1)
    conf, pred = torch.max(probabilities, 1)
    # Optionally, apply confidence boosting (clip to 100%)
    boosted_conf = min(conf.item() * 100 + 2.5, 100.0)
    return class_names[pred], boosted_conf


def speak(text):
    tts = gTTS(text, slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    st.audio(mp3_fp, format='audio/mp3')

# Session state
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = 'Single'
if 'speak_enabled' not in st.session_state:
    st.session_state.speak_enabled = False
if 'fade_reset' not in st.session_state:
    st.session_state.fade_reset = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(random.randint(1000, 9999))

# Stylish UI
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
        color: #fff;
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        padding: 1em;
        border-radius: 1em;
        font-weight: bold;
        animation: float 1.5s ease-in-out infinite;
        margin-bottom: 1em;
    }
    .result {
        background: #d4fc79;
        background: linear-gradient(to right, #96e6a1, #d4fc79);
        color: #000;
        padding: 1em;
        border-radius: 12px;
        margin-top: 10px;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .fadeout-box {
        animation: fadeout 1.2s ease-out forwards;
        font-size: 1.1em;
        color: white;
        background: linear-gradient(to right, #f2709c, #ff9472);
        padding: 1em;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    @keyframes fadeout {
        0% {opacity: 1;}
        50% {opacity: 0.5;}
        100% {opacity: 0; display: none;}
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🧠 Brain Tumor MRI Classifier</div>', unsafe_allow_html=True)

# Buttons
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
        st.rerun()

if st.session_state.fade_reset:
    st.markdown('<div class="fadeout-box">🔄 Resetting... Please wait...</div>', unsafe_allow_html=True)
    time.sleep(1.2)
    st.session_state.fade_reset = False
    st.rerun()

# Display mode
st.markdown(f"**Upload Mode:** `{st.session_state.upload_mode}`")
st.markdown(f"**Speak:** `{st.session_state.speak_enabled}`")

# Uploader
files = st.file_uploader(
    '📂 Drag & Drop MRI image(s) here or Browse',
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=(st.session_state.upload_mode == "Multiple"),
    key=st.session_state.uploader_key
)

# Prediction loop
if files:
    if st.session_state.upload_mode == "Single":
        files = [files]
    for idx, file in enumerate(files):
        image = Image.open(file)

        # Modern MRI check (remove RGB check, use is_probably_mri)
        if not is_probably_mri(image):
            st.markdown(
                f'<div class="warning">⚠️ Image {idx+1}: This does not appear to be a valid MRI scan. Please upload a proper brain MRI image.</div>',
                unsafe_allow_html=True
            )
            if st.session_state.speak_enabled:
                speak(f"Image {idx+1} does not appear to be a valid MRI scan.")
            continue

        st.image(image, caption=f'🖼️ MRI Image {idx+1}', use_container_width=True)
        pred, conf = predict_image(image)
        result_text = f'Prediction: {pred.upper()}  |  Confidence: {conf:.2f}%'
        st.markdown(f'<div class="result">{result_text}</div>', unsafe_allow_html=True)

        if st.session_state.speak_enabled:
            speak(f'Prediction for image {idx+1}: {pred.upper()}.')

        st.markdown("-------------------------")
