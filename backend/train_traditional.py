# backend/train_lstm.py

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from utils.evaluation import evaluate_classification, print_evaluation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

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


def build_lstm_model(
    vocab_size: int,
    max_len: int,
    dropout1: float = 0.3,
    dropout2: float = 0.2,
    optimizer: str = "adam",
    use_batchnorm: bool = False,
):
    layers = [
        Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        LSTM(128, return_sequences=False),
    ]

    if use_batchnorm:
        layers.append(BatchNormalization())

    layers.extend([
        Dropout(dropout1),
        Dense(64, activation="relu"),
    ])

    if use_batchnorm:
        layers.append(BatchNormalization())

    layers.extend([
        Dropout(dropout2),
        Dense(1, activation="sigmoid"),
    ])

    model = Sequential(layers)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_training_plots(history, experiment_name: str):
    # Accuracy plot
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Accuracy - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{experiment_name}_accuracy.png"))
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Loss - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{experiment_name}_loss.png"))
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["generated"],
        test_size=0.2,
        random_state=42,
        stratify=df["generated"],
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    experiments = [
        {
            "name": "LSTM_baseline",
            "dropout1": 0.3,
            "dropout2": 0.2,
            "optimizer": "adam",
            "use_batchnorm": False,
        },
        {
            "name": "LSTM_more_dropout",
            "dropout1": 0.5,
            "dropout2": 0.3,
            "optimizer": "adam",
            "use_batchnorm": False,
        },
        {
            "name": "LSTM_batchnorm_rmsprop",
            "dropout1": 0.3,
            "dropout2": 0.2,
            "optimizer": "rmsprop",
            "use_batchnorm": True,
        },
    ]

    results = []
    best_model = None
    best_f1 = -1.0
    best_experiment = None

    for exp in experiments:
        print(f"\n===== Running {exp['name']} =====")

        model = build_lstm_model(
            vocab_size=vocab_size,
            max_len=MAX_LEN,
            dropout1=exp["dropout1"],
            dropout2=exp["dropout2"],
            optimizer=exp["optimizer"],
            use_batchnorm=exp["use_batchnorm"],
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train_pad,
            y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1,
        )

        save_training_plots(history, exp["name"])

        y_probs = model.predict(X_test_pad, verbose=0).flatten()
        y_pred = (y_probs >= 0.5).astype(int)

        metrics = evaluate_classification(
            y_test,
            y_pred,
            model_name=exp["name"],
        )
        print_evaluation(metrics)

        results.append({
            "model_name": exp["name"],
            "dropout1": exp["dropout1"],
            "dropout2": exp["dropout2"],
            "optimizer": exp["optimizer"],
            "use_batchnorm": exp["use_batchnorm"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
        })

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = model
            best_experiment = exp

    if best_model is None or best_experiment is None:
        raise RuntimeError("No experiment completed successfully.")

    best_model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "threshold": 0.5,
        "selected_experiment": best_experiment["name"],
        "dropout1": best_experiment["dropout1"],
        "dropout2": best_experiment["dropout2"],
        "optimizer": best_experiment["optimizer"],
        "use_batchnorm": best_experiment["use_batchnorm"],
        "all_results": results,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n===== FINAL COMPARISON =====")
    for r in results:
        print(
            f"{r['model_name']} | "
            f"Accuracy={r['accuracy']:.4f} | "
            f"Precision={r['precision']:.4f} | "
            f"Recall={r['recall']:.4f} | "
            f"F1={r['f1_score']:.4f}"
        )

    print(f"\nBest experiment selected: {best_experiment['name']}")
    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")
    print(f"Saved config to: {CONFIG_PATH}")
    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()