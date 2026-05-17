import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import gdown

# Google Drive file ID for best.pt
MODEL_FILE_ID = "1_UoqgdV0yg2CcRt4daZGWP7sKPi9Cw5b"
MODEL_PATH = "best.pt"

# Page settings
st.set_page_config(page_title="Apple Yield Prediction", page_icon="🍎", layout="wide")

st.title("🍎 Apple Detection and Yield Prediction")
st.write(
    "Upload multiple orchard images. The model will detect apples in each image "
    "and estimate the total yield."
)


@st.cache_resource
def load_model():
    # Download model from Google Drive if not already present
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        with st.spinner("Downloading model... This may take a minute."):
            gdown.download(url, MODEL_PATH, quiet=False)

    return YOLO(MODEL_PATH)


model = load_model()

# Multiple file uploader
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    total_apples = 0

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name)

        apple_count = len(results[0].boxes)
        total_apples += apple_count

        result_image = results[0].plot()

        st.subheader(f"Image {idx}: {uploaded_file.name}")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

    st.metric("Estimated Total Weight", f"{total_weight_kg:.2f} kg") 