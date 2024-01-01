from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('/abc.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction == 'negative':
        return render_template('heart_attack.html', prediction_text='You are not likely to have a heart attack')
    else:
        return render_template('heart_attack.html', prediction_text='You are likely to have a heart attack')
    
app.run(debug=True)