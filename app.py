
from flask import Flask, render_template, request
from joblib import load
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Memuat model dan CountVectorizer
model = load('logistic_regression_model.pkl')
cv = load('count_vectorizer.pkl')

ps = PorterStemmer()

# Pra-pemrosesan teks
def preprocess_text(text):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text.lower())
    review_words = review.split()
    review_words = [ps.stem(word) for word in review_words if not word in set(stopwords.words('english'))]
    return ' '.join(review_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        cleaned_review = preprocess_text(review)
        vectorized_review = cv.transform([cleaned_review]).toarray()
        prediction = model.predict(vectorized_review)

        sentiment = "Positive" if prediction == 1 else "Negative"

        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)