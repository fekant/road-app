import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

st.title("Traffic Sign Classifier (.tflite)")

@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="traffic_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

interpreter = load_tflite_model()

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((30, 30))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.expand_dims(np.array(image), axis=0).astype(np.float32)
    prediction = predict_image(interpreter, img_array)
    st.success(f"Predicted Class ID: {prediction}")
