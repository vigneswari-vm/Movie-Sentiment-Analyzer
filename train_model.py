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

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample data — you can replace with your own dataset
data = {
    'review': [
        "I love this movie!", 
        "This film was terrible", 
        "An excellent experience", 
        "Worst acting ever", 
        "A wonderful storyline", 
        "I hated the direction"
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Text preprocessing
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save vectorizer and model together
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model trained and saved as model.pkl")
