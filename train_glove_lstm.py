# backend/train_glove_lstm.py

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
GLOVE_PATH = os.path.join(BASE_DIR, "embeddings", "glove.6B.100d.txt")

MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

MODEL_PATH = os.path.join(MODEL_DIR, "glove_lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "glove_lstm_tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "glove_lstm_config.json")

ACCURACY_PLOT_PATH = os.path.join(PLOTS_DIR, "glove_lstm_accuracy.png")
LOSS_PLOT_PATH = os.path.join(PLOTS_DIR, "glove_lstm_loss.png")
CM_PLOT_PATH = os.path.join(PLOTS_DIR, "glove_lstm_confusion_matrix.png")

MAX_WORDS = 20000
MAX_LEN = 200
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5
THRESHOLD = 0.5


def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'generated' columns.")

    df = df[["text", "generated"]].dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["generated"] = df["generated"].astype(int)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"Dataset loaded. Total rows: {len(df)}")
    print("Class distribution:")
    print(df["generated"].value_counts())

    return df


def load_glove_embeddings(glove_path):
    print("Loading GloVe embeddings...")
    embeddings_index = {}

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    print(f"Loaded {len(embeddings_index)} GloVe word vectors.")
    return embeddings_index


def build_embedding_matrix(tokenizer, embeddings_index, vocab_size):
    print("Building embedding matrix...")
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    matched_words = 0

    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            matched_words += 1

    print(f"Matched words: {matched_words}/{vocab_size}")
    return embedding_matrix


def build_model(vocab_size, embedding_matrix):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_LEN,
            trainable=False
        ),
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


def save_training_plots(history):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("GloVe-LSTM Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("GloVe-LSTM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    print(f"Saved accuracy plot: {ACCURACY_PLOT_PATH}")
    print(f"Saved loss plot: {LOSS_PLOT_PATH}")


def save_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
    disp.plot()
    plt.title("GloVe-LSTM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH)
    plt.close()
    print(f"Saved confusion matrix plot: {CM_PLOT_PATH}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    if not os.path.exists(GLOVE_PATH):
        raise FileNotFoundError(f"GloVe file not found at: {GLOVE_PATH}")

    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["generated"],
        test_size=0.2,
        random_state=42,
        stratify=df["generated"]
    )

    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    embeddings_index = load_glove_embeddings(GLOVE_PATH)
    embedding_matrix = build_embedding_matrix(tokenizer, embeddings_index, vocab_size)

    model = build_model(vocab_size, embedding_matrix)
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    print("Training GloVe-LSTM model...")
    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Evaluating model...")
    y_probs = model.predict(X_test_pad).flatten()
    y_pred = (y_probs >= THRESHOLD).astype(int)

    print("\n===== Classification Report: GloVe-LSTM =====\n")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"], zero_division=0))

    save_training_plots(history)
    save_confusion_matrix_plot(y_test, y_pred)

    print("Saving model...")
    model.save(MODEL_PATH)

    print("Saving tokenizer...")
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "threshold": THRESHOLD,
        "trainable": False,
        "model_name": "glove_lstm"
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")
    print(f"Saved config to: {CONFIG_PATH}")
    print("GloVe-LSTM training completed successfully.")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()