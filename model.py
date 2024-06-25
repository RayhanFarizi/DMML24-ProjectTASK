import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from joblib import dump

nltk.download('stopwords')

# Memuat dataset
df = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')

# Pra-pemrosesan teks
corpus = []
ps = PorterStemmer()

for i in range(0, len(df)):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['Review'][i].lower())
    review_words = review.split()
    review_words = [ps.stem(word) for word in review_words if not word in set(stopwords.words('english'))]
    review = ' '.join(review_words)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df['Liked'].values

# Melatih model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Menyimpan model dan CountVectorizer
dump(model, 'logistic_regression_model.pkl')
dump(cv, 'count_vectorizer.pkl')