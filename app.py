from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# load model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['f1']),
            float(request.form['f2']),
            float(request.form['f3']),
            float(request.form['f4']),
            float(request.form['f5']),
            float(request.form['f6']),
            float(request.form['f7']),
            float(request.form['f8']),
            float(request.form['f9']),
            float(request.form['f10']),
            float(request.form['f11']),
            float(request.form['f12']),
            float(request.form['f13'])
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)

        return render_template(
            'index.html',
            prediction_text=f"Predicted Price: {prediction[0]:.2f}"
        )

    except:
        return render_template(
            'index.html',
            prediction_text="⚠️ Invalid input. Please enter valid numbers."
        )

if __name__ == "__main__":
    app.run(debug=True)