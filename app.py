# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('model/sentiment_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    data = vectorizer.transform([review])
    prediction = model.predict(data)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
