import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
sc = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('Gb_classifier.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model.predict( sc.transform(final_features) )
    
    return render_template('result.html', prediction = pred)

if __name__ == "__main__":
    app.run(debug=True)
