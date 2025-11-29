import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model
model = tf.keras.models.load_model("model/garbage_classifier_cnn_final.h5")

labels = ['battery','biological','brown-glass','cardboard',
          'clothes','green-glass','metal','paper',
          'plastic','shoes','trash','white-glass']

st.title("♻ Garbage Classification AI")
st.write("Upload an image and the model will classify the type of waste.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = load_img(uploaded, target_size=(128,128))
    st.image(img, caption="Uploaded Image", width=300)

    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = labels[np.argmax(model.predict(img))]

    st.success(f"🔍 Prediction: **{prediction.upper()}**")
