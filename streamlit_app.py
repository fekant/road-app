import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

@st.cache_resource
def load_model_file():
    return load_model("traffic_classifier.h5")

model = load_model_file()

st.title("Traffic Sign Classifier")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((30, 30))
    st.image(img, caption="Input Image", use_column_width=True)
    img_array = np.expand_dims(np.array(img), axis=0)
    prediction = np.argmax(model.predict(img_array))
    st.success(f"Predicted Class ID: {prediction}")
