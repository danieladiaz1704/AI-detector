import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.evaluation import evaluate_classification, print_evaluation


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "traditional_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "traditional_vectorizer.pkl")


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'generated' columns.")

    df = df[["text", "generated"]].dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["generated"] = df["generated"].astype(int)

    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["generated"],
        test_size=0.2,
        random_state=42,
        stratify=df["generated"]
    )

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words=None
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    metrics = evaluate_classification(y_test, y_pred, model_name="Logistic Regression")
    print_evaluation(metrics)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()