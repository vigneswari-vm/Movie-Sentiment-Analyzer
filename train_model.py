# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv('data/IMDB Dataset.csv')
X = df['review']
y = df['sentiment']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# Save the model
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
