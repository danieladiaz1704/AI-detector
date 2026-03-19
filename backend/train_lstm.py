# backend/train_lstm.py

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.evaluation import evaluate_classification, print_evaluation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "lstm_tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "lstm_config.json")

MAX_WORDS = 20000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5


def clean_text(text: str) -> str:
    text = str(text).strip().lower()
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


def build_lstm_model(vocab_size: int, max_len: int):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


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

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model = build_lstm_model(vocab_size, MAX_LEN)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    y_probs = model.predict(X_test_pad).flatten()
    y_pred = (y_probs >= 0.5).astype(int)

    metrics = evaluate_classification(y_test, y_pred, model_name="LSTM")
    print_evaluation(metrics)

    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "threshold": 0.5
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")
    print(f"Saved config to: {CONFIG_PATH}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()