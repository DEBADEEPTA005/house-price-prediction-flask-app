from flask import Flask, request, render_template
import pickle
import numpy as np

# load model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get input values
    features = [
    float(request.form['f1']),   # CRIM
    float(request.form['f2']),   # ZN
    float(request.form['f3']),   # INDUS
    float(request.form['f4']),   # CHAS
    float(request.form['f5']),   # NOX
    float(request.form['f6']),   # RM
    float(request.form['f7']),   # AGE
    float(request.form['f8']),   # DIS
    float(request.form['f9']),   # RAD
    float(request.form['f10']),  # TAX
    float(request.form['f11']),  # PTRATIO
    float(request.form['f12']),  # B
    float(request.form['f13'])   # LSTAT
]
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text=f"Predicted Price: {prediction[0]:.2f}")

if __name__ == "__main__":
    app.run(debug=True)