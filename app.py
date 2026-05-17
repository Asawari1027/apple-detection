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
    "The model will detect apples and estimate the total yield."
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

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Read image
        image = Image.open(uploaded_file).convert("RGB")

        # Save temporarily for YOLO inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # Run detection
        results = model(temp_path)

        # Count detections
        apple_count = len(results[0].boxes)
        total_apples += apple_count

        # Get annotated image
        result_image = results[0].plot()

        # Display results
        st.subheader(f"Image {idx}: {uploaded_file.name}")
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                result_image,
                caption=f"Detected Apples: {apple_count}",
                use_container_width=True
            )

        st.info(f"🍎 Apples detected in this image: {apple_count}")
        st.divider()

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Total yield summary
    st.success(f"🌳 Total Estimated Yield: {total_apples} apples")

    average_weight = st.number_input(
        "Average weight per apple (grams)",
        min_value=1,
        value=180
    )

    total_weight_kg = (total_apples * average_weight) / 1000

    st.metric(
        "Estimated Total Weight",
        f"{total_weight_kg:.2f} kg"
    )
