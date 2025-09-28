import os, json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage

MODEL_PATH = os.path.join("outputs", "checkpoints", "modelB_phase2_best.keras")  # change if your filename differs
LABELS_PATH = "labels.json"
IMG_SIZE = (224, 224)  # must match training input size

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels

def preprocess_pil(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict(img_pil, model, labels):
    arr = preprocess_pil(img_pil)
    probs = model.predict(arr, verbose=0)[0]  # shape: (num_classes,)
    idx = int(np.argmax(probs))
    top_label = labels[idx]
    top_conf = float(probs[idx]) * 100.0
    order = np.argsort(probs)[::-1]
    ranked = [(labels[i], float(probs[i]) * 100.0) for i in order]
    return top_label, top_conf, ranked

st.set_page_config(page_title="Plant Disease Detector (Part B)", page_icon="🌿")
st.title("🌿 Plant Disease Detector — Part B")
st.caption("MobileNetV2 (transfer learning). Upload a tomato leaf image to get a prediction.")

with st.spinner("Loading model…"):
    model, labels = load_model_and_labels()

uploaded = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

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
