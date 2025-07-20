import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import gtts
import tempfile
import os
import requests

# ───── App Config ───── #
st.set_page_config(page_title="🧠 MRI Classifier", layout="centered", page_icon="🧠")
st.markdown(
    "<h1 style='text-align:center;color:#00ffff;'>MRI Brain Tumor Classifier</h1>",
    unsafe_allow_html=True
)

# ───── Upload Mode + Options ───── #
col1, col2, col3, col4 = st.columns(4)
upload_mode = col1.toggle("📁 Multiple Images", value=False)
speak_enabled = col2.toggle("🔈 Toggle Speak", value=True)
col3.button("🧹 Clear / Reset", on_click=st.experimental_rerun)
col4.markdown("")

# ───── File Uploader ───── #
st.markdown("### 📤 Drag & Drop MRI image(s) here or Browse")
uploaded_files = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=upload_mode,
    label_visibility="collapsed"
)

# ───── Model Definition ───── #
class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 56 * 56, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    gdrive_url = "https://drive.google.com/uc?id=1kb6mE0dgcftsHxZjgQxhFZF3QjzvpJgs"
    model_path = "model.pt"
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(gdrive_url).content)
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def is_mri_image(tensor):
    avg_channels = tensor.mean(dim=[1, 2])
    diff = torch.abs(avg_channels[0] - avg_channels[1]) + torch.abs(avg_channels[1] - avg_channels[2])
    return diff < 0.2

def speak(text):
    tts = gtts.gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

def predict(image):
    tensor = transform(image).unsqueeze(0)
    if not is_mri_image(tensor[0]):
        st.warning("⚠️ This image may not be a valid MRI scan.")
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        return class_names[pred.item()], conf.item()

# ───── Inference UI ───── #
if uploaded_files:
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for idx, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        st.image(image, caption=f"🖼️ MRI Image {idx+1}", use_column_width=True)

        pred, conf = predict(image)
        msg = f"Prediction: {pred.upper()} | Confidence: {conf * 100:.2f}%"

        st.markdown(f"<div style='background-color:#d1ffd6;padding:10px;border-radius:10px;font-weight:bold;'>{msg}</div>", unsafe_allow_html=True)

        if speak_enabled:
            speak(msg)
