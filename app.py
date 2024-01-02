from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template('heart_attack.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    # Ensure that the form fields in heart_attack.html have the correct names
    feature_names = ["age","impluse","pressurehight","pressurelow","glucose","kcm","troponin"]

    # Retrieve feature values from the form
    try:
        # Convert input values to floats
        int_features = [float(request.form.get(name, 0)) for name in feature_names]
    except ValueError:
        return render_template('heart_attack.html', pred='Invalid input. Please enter valid numeric values.')

    # Print for debugging
    print("Input Features:", int_features)

    # Ensure that at least one feature is present
    if not any(int_features):
        return render_template('heart_attack.html', pred='Invalid input. Please provide values for at least one feature.')

    # Reshape the features into a 2D array
    final_features = np.array(int_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)

    # Print for debugging
    print("Prediction:", prediction)

    if prediction[0] == 'negative':
        return render_template('heart_attack.html', pred='You are not likely to have a heart attack')
    else:
        return render_template('heart_attack.html', pred='You are likely to have a heart attack')
    
if __name__ == "__main__":
    app.run()
