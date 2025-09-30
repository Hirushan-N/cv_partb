import os, json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage

# ==============================
# NEW: public test images folder
# ==============================
TEST_IMAGES_URL = "https://drive.google.com/drive/folders/1A4QN4dRvQT4RshBT2J-KKjc3OXCkVET2?usp=sharing"
# Optional: the folder ID if you want to show a gdown command
TEST_FOLDER_ID  = "1A4QN4dRvQT4RshBT2J-KKjc3OXCkVET2"

MODEL_PATH  = os.path.join("outputs", "checkpoints", "modelB_phase2_best.keras")
LABELS_PATH = "labels.json"
IMG_SIZE    = (224, 224)
MIN_BYTES   = 1_000_000

st.set_page_config(page_title="Plant Disease Detector (Part B)", page_icon="🌿")

# ---------------------------------------
# NEW: Sidebar helper to download samples
# ---------------------------------------
with st.sidebar:
    st.header("🧪 Get Sample Test Images")
    st.write("Use these tomato leaf images to try the app quickly.")
    st.link_button("Download sample images (Google Drive)", TEST_IMAGES_URL)
    with st.expander("Use gdown (advanced)"):
        st.markdown(
            "You can download the entire folder locally with **gdown**:\n\n"
            "```bash\n"
            f"pip install gdown\n"
            f"gdown --folder {TEST_FOLDER_ID} -O sample_tests\n"
            "```"
        )
    st.divider()

st.title("🌿 Plant Disease Detector")
st.caption("MobileNetV2 (transfer learning). Upload a leaf image to get a prediction.")

# ---------------------------------------
# Model file checks / loading
# ---------------------------------------
def ensure_dirs():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def ensure_model_local():
    """
    Ensure the model file exists locally.
    If missing or too small, prompt user to upload the .keras file (one-time).
    """
    ensure_dirs()
    needs_model = (not os.path.exists(MODEL_PATH)) or (os.path.getsize(MODEL_PATH) < MIN_BYTES)

    if needs_model:
        st.warning(
            "Model file is missing on this server. "
            "Please upload your **.keras** weights (e.g., modelB_phase2_best.keras)."
        )
        uploaded_model = st.file_uploader("Upload model file (.keras)", type=["keras", "h5", "pb"], key="model_uploader")
        if uploaded_model is not None:
            try:
                with open(MODEL_PATH, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                st.success(f"Model saved to {MODEL_PATH}.")
            except Exception as e:
                st.error(f"Failed to save uploaded model: {e}")
                st.stop()

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < MIN_BYTES:
        st.error(
            f"Model not available at {MODEL_PATH}. "
            "Commit the model via Git LFS **or** upload it above."
        )
        st.stop()

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    ensure_model_local()

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"labels.json not found at: {LABELS_PATH}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        raise ValueError(f"Could not load model from {MODEL_PATH}: {e}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # quick sanity check on output size vs labels
    try:
        dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
        dummy = tf.keras.applications.mobilenet_v2.preprocess_input(dummy)
        out = model.predict(dummy, verbose=0)
        if out.shape[-1] != len(labels):
            raise ValueError(
                f"Model output classes ({out.shape[-1]}) != labels length ({len(labels)}). "
                "Ensure labels.json order matches training class_names."
            )
    except Exception:
        pass

    return model, labels

def preprocess_pil(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img_pil, model, labels):
    arr = preprocess_pil(img_pil)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    top_label = labels[idx]
    top_conf = float(probs[idx]) * 100.0
    order = np.argsort(probs)[::-1]
    ranked = [(labels[i], float(probs[i]) * 100.0) for i in order]
    return top_label, top_conf, ranked

# ---------------------------------------
# Load model + labels
# ---------------------------------------
try:
    with st.spinner("Loading model…"):
        model, labels = load_model_and_labels()
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------
# Uploader + Inference
# ---------------------------------------
uploaded = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="image_uploader")

if uploaded is not None:
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)

        with st.spinner("Analyzing…"):
            top_label, top_conf, ranked = predict(img, model, labels)

        st.subheader("Result")
        st.write(f"**Prediction:** {top_label}")
        st.write(f"**Confidence:** {top_conf:.2f}%")

        st.markdown("### All class probabilities")
        for name, conf in ranked:
            st.write(f"- {name}: {conf:.2f}%")

        if top_conf < 50:
            st.info("Low confidence — try a clearer image or better lighting.")
    except Exception as e:
        st.error(f"Failed to process the image: {e}")

st.markdown("---")
st.caption("Part B — Computer Vision Coursework • Streamlit + TensorFlow (MobileNetV2)")
