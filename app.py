# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model 
MODEL = joblib.load('spam_detection_model.pkl')

# Create an instance of CountVectorizer
vectorizer = joblib.load('vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        # print("Email text:",email_text)
        # Vectorize the input text
        input_text_vectorized = vectorizer.transform([email_text])
        # Make the prediction
        prediction = MODEL.predict(input_text_vectorized)
        # print("Prediction:",prediction)

        # Map prediction back to 'ham' or 'spam'
        result = 'ham' if prediction[0] == 0 else 'spam'

        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
