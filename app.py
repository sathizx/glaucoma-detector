import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Glaucoma Detection", layout="wide")
st.title("Glaucoma Detection from Retinal Images")
st.header("Glaucoma Detection")
st.write("Upload a retinal image for analysis:")

uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None

# Train Model function
def train_model():
    try:
        st.write("📈 Training model... please wait.")

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_data = datagen.flow_from_directory(
            'dataset',
            target_size=(128, 128),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        val_data = datagen.flow_from_directory(
            'dataset',
            target_size=(128, 128),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_data, validation_data=val_data, epochs=5)

        os.makedirs("saved_models", exist_ok=True)
        model.save('saved_models/glaucoma_model.h5')

        st.session_state.model = model
        st.session_state.model_trained = True

        st.sidebar.success("✅ Model trained and loaded!")

    except Exception as e:
        st.sidebar.error(f"Training failed: {str(e)}")

# Sidebar options
st.sidebar.title("Options")

if st.sidebar.button("Train Model"):
    train_model()

if st.sidebar.button("Load Pre-trained Model"):
    try:
        st.session_state.model = load_model('saved_models/glaucoma_model.h5')
        st.session_state.model_trained = True
        st.sidebar.success("Model loaded!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {str(e)}")

# Prediction
if uploaded_file is not None and st.session_state.model_trained:
    try:
        image = Image.open(uploaded_file).resize((128, 128))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        pred_prob = st.session_state.model.predict(image_array)[0][0]
        pred = 1 if pred_prob >= 0.5 else 0

        if pred == 0:
            st.error("🩺 Glaucoma Detected")
            st.warning("Please consult an ophthalmologist for further evaluation.")
        else:
            st.success("✅ No Glaucoma Detected")
            st.info("Regular eye checkups are recommended for prevention.")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Glaucoma', 'Normal'], [1 - pred_prob, pred_prob], color=['red', 'green'])
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Confidence')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
elif uploaded_file is not None:
    st.warning("⚠️ Please train or load a model first using the sidebar options.")
