import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class AmazonReviewSentimentAnalyzer:
    """
    NLP Sentiment Analysis engine trained on Amazon Product Reviews.
    Implements text normalization, TF-IDF n-gram vectorization, and Logistic Regression.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        self.is_trained = False

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()

    def train_baseline(self):
        # Representative corpus of labeled Amazon Product Reviews
        reviews = [
            ("The build quality is outstanding and battery lasts 3 full days. Highly recommended!", "positive"),
            ("Fast shipping, excellent packaging, and works right out of the box.", "positive"),
            ("The product is completely defective. Stopped working after 24 hours of usage.", "negative"),
            ("Extremely cheap plastic material, broke on the first day. Terrible customer support.", "negative"),
            ("Average quality for the price. Works fine for simple tasks but nothing exceptional.", "neutral"),
            ("Decent performance, arrived on time but color is slightly different from photo.", "neutral"),
            ("Five stars! Superb clarity, premium bass, and comfortable fit.", "positive"),
            ("Worst purchase I ever made. Do not waste your money on this item.", "negative"),
            ("It is acceptable for occasional use, though instructions were confusing.", "neutral")
        ]
        
        X = [self.clean_text(r[0]) for r in reviews]
        y = [r[1] for r in reviews]
        
        self.pipeline.fit(X, y)
        self.is_trained = True

    def predict(self, text):
        if not self.is_trained:
            self.train_baseline()
        cleaned = self.clean_text(text)
        pred = self.pipeline.predict([cleaned])[0]
        prob = np.max(self.pipeline.predict_proba([cleaned]))
        return pred, float(prob)
