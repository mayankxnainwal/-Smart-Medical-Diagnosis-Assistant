import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_text(csv_path, output_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    X, y = df['text'].astype(str), df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    print(classification_report(y_val, preds))

    joblib.dump(pipe, output_path)
    print(f"✅ Saved text model to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    train_text(args.csv, args.output)
