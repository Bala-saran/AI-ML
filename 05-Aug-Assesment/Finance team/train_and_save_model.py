import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Sample spam dataset
data = {
    'text': [
        "Congratulations! You've won a free ticket",
        "You have won ₹1,000,000! Call now!",
        "Hey, are we meeting today?",
        "Reminder: your appointment is tomorrow.",
        "URGENT! Your account will be locked!"
    ],
    'label': [1, 1, 0, 0, 1]  # 1 = spam, 0 = not spam
}
df = pd.DataFrame(data)

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ model.pkl and vectorizer.pkl saved.")
