"""
Streamlit UI for Cats vs Dogs Classifier.
"""

import requests
import streamlit as st
from PIL import Image

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="centered",
)

st.title("🐾 Cats vs Dogs Classifier")
st.markdown("Upload an image to find out whether it's a **cat** or a **dog**.")

# --- Sidebar: API status ---
with st.sidebar:
    st.header("API Status")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        data = resp.json()
        if data.get("status") == "healthy":
            st.success("API: Healthy ✅")
            st.success("Model: Loaded ✅")
        else:
            st.warning("API: Degraded ⚠️")
            st.error("Model: Not loaded ❌")
    except Exception:
        st.error("API: Unreachable ❌")
    st.markdown("---")
    st.caption("API endpoint: `http://localhost:8000`")

# --- Main: upload & predict ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify", use_container_width=True):
        with st.spinner("Classifying..."):
            try:
                file_bytes = uploaded_file.getvalue()

                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    predicted_class = result["class"]
                    confidence = result["confidence"]
                    dog_prob = result["dog_probability"]
                    cat_prob = result["cat_probability"]

                    # Result banner
                    emoji = "🐶" if predicted_class == "dog" else "🐱"
                    st.markdown("---")
                    st.subheader(f"Prediction: {emoji} **{predicted_class.upper()}**")

                    # Confidence bar
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                    st.progress(confidence)

                    # Probability breakdown
                    st.markdown("#### Probability Breakdown")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🐱 Cat", f"{cat_prob * 100:.1f}%")
                        st.progress(cat_prob)
                    with col2:
                        st.metric("🐶 Dog", f"{dog_prob * 100:.1f}%")
                        st.progress(dog_prob)

                elif response.status_code == 503:
                    st.error("Model is not loaded. Please check the API.")
                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure it is running on port 8000.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
