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
        return render_template('heart_attack.html', pred='You are Safe',pred2=" However, when it comes to real-life scenarios, it's crucial to prioritize heart health. Regular exercise, a balanced diet, and routine check-ups contribute to a heart-healthy lifestyle. Remember, early detection and preventive measures play a pivotal role. Explore our site on heart attack prediction for valuable insights and tips to safeguard your cardiovascular well-being. Stay informed, stay active, and prioritize your heart health for a longer, happier life. Thanks for using our service, keep chechking your health. Remember, a heart-healthy lifestyle is a journey, not a destination. By making informed choices and adopting positive habits, you can significantly reduce your risk of a heart attack and enjoy a longer, healthier life")
    else:
        return render_template('heart_attack.html', pred='You are likely to have an heart attack',pred2="Contact a specialist or a Doctor Immeiately. Your health is at severe risk. Your well-being is our priority. While we're not medical professionals, we encourage you to be mindful of potential heart health concerns. Recognizing symptoms like chest pain, shortness of breath, or fatigue is crucial. However, only a qualified healthcare provider can accurately assess your risk of a heart attack. Regular check-ups and a healthy lifestyle are key preventive measures. Remember, this note is not a substitute for professional medical advice. If you have concerns, consult a healthcare professional promptly. Stay proactive in maintaining a heart-healthy lifestyle for a longer, happier life. Factors contributing to increased risk include lifestyle choices, family medical history, and specific health indicators. It is imperative for those identified to consider consulting with healthcare professionals promptly. Early detection and appropriate lifestyle modifications can play a pivotal role in mitigating the risk of heart-related issues. Thanks for using our service, keep chechking your health.")
    
if __name__ == "__main__":
    app.run(debug=False)
