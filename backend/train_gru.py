"""
backend/train_gru.py

Experiment 1 (supplementary) — GRU with 3 hyperparameter configurations:
  1. GRU_baseline    — single GRU layer, adam
  2. GRU_stacked     — two stacked GRU layers, adam
  3. GRU_bidirectional — Bidirectional GRU, rmsprop

Saves the best model (by F1) for use by the API.
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.evaluation import (
    evaluate_classification,
    print_evaluation,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, GRU, Bidirectional,
    Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH     = os.path.join(MODEL_DIR, "gru_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "gru_tokenizer.pkl")
CONFIG_PATH    = os.path.join(MODEL_DIR, "gru_config.json")

MAX_WORDS    = 20_000
MAX_LEN      = 200
EMBEDDING_DIM = 128
BATCH_SIZE   = 32
EPOCHS       = 5


# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_data():
    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'generated' columns.")
    df = df[["text", "generated"]].dropna()
    df["text"]      = df["text"].astype(str).apply(clean_text)
    df["generated"] = df["generated"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────
def build_baseline_gru(vocab_size: int) -> Sequential:
    """Single GRU layer — simplest configuration."""
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        GRU(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_stacked_gru(vocab_size: int) -> Sequential:
    """Two stacked GRU layers with BatchNormalization."""
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        GRU(128, return_sequences=True),
        BatchNormalization(),
        GRU(64, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_bidirectional_gru(vocab_size: int) -> Sequential:
    """Bidirectional GRU — reads the sequence in both directions."""
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        Bidirectional(GRU(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


EXPERIMENTS = [
    {"name": "GRU_baseline",       "builder": build_baseline_gru,       "description": "Single GRU, adam, dropout 0.3/0.2"},
    {"name": "GRU_stacked",        "builder": build_stacked_gru,        "description": "Stacked GRU ×2, adam, BatchNorm"},
    {"name": "GRU_bidirectional",  "builder": build_bidirectional_gru,  "description": "Bidirectional GRU, rmsprop"},
]


# ─────────────────────────────────────────────────────────────
def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"], df["generated"],
        test_size=0.2, random_state=42, stratify=df["generated"],
    )
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    # ── Tokenise ────────────────────────────────────────────
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_raw)

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_train_raw),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_test_raw),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    results     = []
    best_f1     = -1.0
    best_model  = None
    best_exp    = None

    for exp in EXPERIMENTS:
        print(f"\n{'='*55}")
        print(f"  Running: {exp['name']}")
        print(f"  Config : {exp['description']}")
        print(f"{'='*55}")

        model = exp["builder"](vocab_size)
        model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )

        history = model.fit(
            X_train_pad, y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )

        # ── Save training curves ─────────────────────────────
        plot_training_curves(history.history, exp["name"], RESULTS_DIR)

        # ── Evaluate ─────────────────────────────────────────
        y_proba = model.predict(X_test_pad, verbose=0).flatten()
        y_pred  = (y_proba >= 0.5).astype(int)

        metrics = evaluate_classification(y_test, y_pred, exp["name"], y_proba=y_proba)
        print_evaluation(metrics)

        plot_confusion_matrix(y_test, y_pred, exp["name"], RESULTS_DIR)
        plot_roc_curve(y_test, y_proba, exp["name"], RESULTS_DIR)

        results.append({
            "model_name"   : exp["name"],
            "description"  : exp["description"],
            "accuracy"     : metrics["accuracy"],
            "precision"    : metrics["precision"],
            "recall"       : metrics["recall"],
            "f1_score"     : metrics["f1_score"],
            "roc_auc"      : metrics["roc_auc"],
            "fpr"          : metrics["fpr"],
            "tpr"          : metrics["tpr"],
        })

        if metrics["f1_score"] > best_f1:
            best_f1    = metrics["f1_score"]
            best_model = model
            best_exp   = exp

    # ── Persist ──────────────────────────────────────────────
    best_model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_words"          : MAX_WORDS,
        "max_len"            : MAX_LEN,
        "embedding_dim"      : EMBEDDING_DIM,
        "threshold"          : 0.5,
        "selected_experiment": best_exp["name"],
        "best_f1"            : best_f1,
        "all_results"        : [
            {k: v for k, v in r.items() if k not in ("fpr", "tpr")}
            for r in results
        ],
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # ── Final summary ─────────────────────────────────────────
    print("\n===== GRU EXPERIMENT COMPARISON =====")
    print(f"{'Model':<25} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 60)
    for r in results:
        auc = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "  N/A "
        print(
            f"{r['model_name']:<25} "
            f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} {r['f1_score']:>7.4f} {auc:>7}"
        )

    print(f"\nBest: {best_exp['name']}  (F1={best_f1:.4f})")
    print(f"Saved model     → {MODEL_PATH}")
    print(f"Saved tokenizer → {TOKENIZER_PATH}")
    print(f"Saved config    → {CONFIG_PATH}")
    print(f"Plots saved     → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()