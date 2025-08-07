# app.py

import joblib

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Message to test
text = ["You have won ₹1,000,000! Call now!"]  # change this text to test different messages

# Transform the input text using the same vectorizer
X_input = vectorizer.transform(text)

# Predict using the model
prediction = model.predict(X_input)

# Output result
if prediction[0] == 1:
    print("🚨 Spam detected!")
else:
    print("✅ Not spam.")
