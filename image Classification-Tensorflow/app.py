import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

st.title("CIFAR-10 Image Classification")

@st.cache_resource
def load_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model = load_model()

# Option to train the model
if st.button("Train Model"):
    st.write("Training on CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    model.save("cifar10_model.h5")
    st.success("Training completed and model saved!")

# Option to load a pre-trained model
if st.button("Load Pretrained Model"):
    try:
        model = tf.keras.models.load_model("cifar10_model.h5")
        st.success("Model loaded successfully!")
    except:
        st.error("No saved model found. Please train first.")

# Image upload for prediction
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = tf.nn.softmax(predictions)[0][predicted_class].numpy()

    st.image(image, caption=f"Predicted: {classes[predicted_class]} ({confidence:.2f} confidence)")
