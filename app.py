from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def hellow_world():
    return render_template('heart_attack.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)
    print(prediction)
    # if prediction == ['negative']:
    #     return render_template('heart_attack.html', pred='You are not likely to have a heart attack')
    # else:
    #     return render_template('heart_attack.html', pred='You are likely to have a heart attack')

if __name__ == "__main__":
    app.run()