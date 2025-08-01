import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample data
data = {
    'text': [
        'Win money now',
        'Limited offer just for you',
        'Hi, how are you?',
        'Call me tomorrow',
        'Free tickets available',
        'Congratulations, you won!',
        'Are you coming to the party?',
        'Let’s grab lunch today',
        'Earn extra cash fast',
        'Meeting at 10 am'
    ],
    'label': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}
df = pd.DataFrame(data)

# Train the model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved!")
