from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open(r"C:/Users/balas/OneDrive/Downloads/TAXI WEB/data/taxi_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        input_data = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4'])
        ]
        
        # Convert to numpy array and reshape for model
        features = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Render result page
        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
