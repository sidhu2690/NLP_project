import re
import nltk
import joblib
from flask import Flask, render_template, request
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load the model and vectorizer
with open('model/logistic_regression_model.pkl', 'rb') as file:
    vectorizer, model = joblib.load(file)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def make_logistic_prediction(sentence):
    global vectorizer, model
    sentence = clean_text(sentence)
    sentence = vectorizer.transform([sentence])
    prediction = model.predict(sentence)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction=None, error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction=None, error="No selected file")
        
        if file:
            df = pd.read_csv(file)
            df['clean_sentence'] = df['sentence'].apply(clean_text)

            prediction_results = {}
            for sentence in df['clean_sentence']:
                prediction_results[sentence] = make_logistic_prediction(sentence)
            return render_template('index.html', prediction=prediction_results, error=None)
    else:
        # Handle the GET request method
        return render_template('index.html', prediction=None, error=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
