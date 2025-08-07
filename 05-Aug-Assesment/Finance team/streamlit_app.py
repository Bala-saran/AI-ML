import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Spam Detector", page_icon="📨")
st.title("📨 Spam Message Detector")

user_input = st.text_area("✍️ Enter your message:")

if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        features = vectorizer.transform([user_input])
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("🚨 Spam detected!")
        else:
            st.success("✅ Message is not spam.")
