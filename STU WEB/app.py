from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open(r'C:/Users/balas/OneDrive/Downloads/STU WEB/data/model_performance.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance_percentage'])
        previous_grade = float(request.form['previous_grade_numeric'])

        # Format the input for prediction
        input_data = [[study_hours, attendance, previous_grade]]
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('result.html', prediction=f"Invalid input: {e}")

if __name__ == '__main__':
    app.run(debug=True)
