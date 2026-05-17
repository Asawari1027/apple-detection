import os
import tempfile

import gdown
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Google Drive file ID for best.pt
MODEL_FILE_ID = "1_UoqgdV0yg2CcRt4daZGWP7sKPi9Cw5b"
MODEL_PATH = "best.pt"

# Page configuration
st.set_page_config(
    page_title="Apple Detection and Yield Prediction",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Apple Detection and Yield Prediction")
st.write(
    "Upload one or more orchard images to detect apples, "
    "estimate yield, and calculate revenue in Indian Rupees (₹)."
)


@st.cache_resource
def load_model():
    """Download best.pt from Google Drive if needed and load the YOLO model."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    return YOLO(MODEL_PATH)


# Load model only once
model = load_model()

# Upload images
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    total_apples = 0
    image_counts = []  # Stores (filename, apple_count)

    # Process each uploaded image
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Read image
        image = Image.open(uploaded_file).convert("RGB")

        # Save temporarily for YOLO inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # Run detection
        results = model(temp_path)

        # Count detected apples
        apple_count = len(results[0].boxes)
        total_apples += apple_count
        image_counts.append((uploaded_file.name, apple_count))

        # Get annotated image
        annotated_image = results[0].plot()

        # Display image results
        st.subheader(f"Image {idx}: {uploaded_file.name}")
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                width="stretch"
            )

        with col2:
            st.image(
                annotated_image,
                caption=f"Detected Apples: {apple_count}",
                width="stretch"
            )

        st.info(f"🍎 Apples detected in this image: {apple_count}")
        st.divider()

        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Yield and revenue settings
    st.subheader("⚙️ Yield and Revenue Settings")

    average_weight = st.number_input(
        "Average weight per apple (grams)",
        min_value=1,
        value=180
    )

    price_per_kg = st.number_input(
        "Market price per kg (₹)",
        min_value=0.0,
        value=120.0,
        step=1.0
    )

    # Calculate per-image and total statistics
    per_image_rows = []
    total_weight_kg = 0.0
    total_revenue = 0.0

    for image_name, apple_count in image_counts:
        image_weight_kg = (apple_count * average_weight) / 1000
        image_revenue = image_weight_kg * price_per_kg

        total_weight_kg += image_weight_kg
        total_revenue += image_revenue

        per_image_rows.append({
            "Image": image_name,
            "Apples": apple_count,
            "Yield (kg)": round(image_weight_kg, 2),
            "Revenue (₹)": round(image_revenue, 2)
        })

    # Per-image table
    st.subheader("📊 Per Image Yield and Revenue")
    st.dataframe(per_image_rows, use_container_width=True)

    # Overall summary
    st.subheader("🌳 Overall Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Apples", total_apples)

    with col2:
        st.metric("Total Yield", f"{total_weight_kg:.2f} kg")

    with col3:
        st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
