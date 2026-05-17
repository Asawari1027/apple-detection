import os
import tempfile

import gdown
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Google Drive file ID for best.pt
MODEL_FILE_ID = "1_UoqgdV0yg2CcRt4daZGWP7sKPi9Cw5b"
MODEL_PATH = "best.pt"

# Page settings
st.set_page_config(
    page_title="Apple Detection and Yield Prediction",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Apple Detection and Yield Prediction")
st.write(
    "Upload one or more orchard images. "
    "The model will detect apples, estimate yield, and calculate revenue in Indian Rupees (₹)."
)


@st.cache_resource
def load_model():
    """Download best.pt from Google Drive if needed and load YOLO model."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    return YOLO(MODEL_PATH)


# Load model once and cache it
model = load_model()

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    total_apples = 0
    image_counts = []  # List of tuples: (filename, apple_count)

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Read image
        image = Image.open(uploaded_file).convert("RGB")

        # Save temporarily for YOLO inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
        st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
