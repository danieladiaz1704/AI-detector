"""
backend/train_unsupervised.py

Experiment 2 — Unsupervised Feature Extraction via Autoencoder.

Pipeline:
  1. Vectorise text with TF-IDF  (no labels used for the autoencoder)
  2. Train a deep Autoencoder to learn a compressed representation
  3. Extract the encoder (bottleneck) output as learned features
  4. Train a Logistic Regression classifier on:
       a) raw TF-IDF features  (baseline)
       b) autoencoder-encoded features  (unsupervised transfer)
  5. Compare and report both classifiers
  6. Save the encoder model + vectorizer for optional API use
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from utils.evaluation import evaluate_classification, print_evaluation


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

ENCODER_PATH    = os.path.join(MODEL_DIR, "autoencoder_encoder.keras")
AE_VEC_PATH     = os.path.join(MODEL_DIR, "autoencoder_vectorizer.pkl")
AE_SCALER_PATH  = os.path.join(MODEL_DIR, "autoencoder_scaler.pkl")
AE_CLF_PATH     = os.path.join(MODEL_DIR, "autoencoder_classifier.pkl")
AE_CONFIG_PATH  = os.path.join(MODEL_DIR, "autoencoder_config.json")

# ── Hyperparameters ─────────────────────────────────────────
TFIDF_MAX_FEATURES =  3_000  # kept manageable for dense autoencoder
ENCODING_DIM       = 128      # bottleneck size
HIDDEN_DIMS        = [2048, 512]   # encoder hidden layers (mirrored in decoder)
DROPOUT_RATE       = 0.3
EPOCHS             = 20
BATCH_SIZE         = 256
LEARNING_RATE      = 1e-3


# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="ignore")
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'generated' columns.")
    df = df[["text", "generated"]].dropna()
    df["text"]      = df["text"].astype(str).apply(clean_text)
    df["generated"] = df["generated"].astype(int)

    # Cap at 50k to avoid MemoryError on dense conversion
    SAMPLE_SIZE = 50_000
    if len(df) > SAMPLE_SIZE:
        half     = SAMPLE_SIZE // 2
        human_df = df[df["generated"] == 0].sample(n=half, random_state=42)
        ai_df    = df[df["generated"] == 1].sample(n=half, random_state=42)
        df       = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Rows used: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────
# Autoencoder builder
# ─────────────────────────────────────────────────────────────
def build_autoencoder(input_dim: int, hidden_dims: list, encoding_dim: int, dropout: float):
    """
    Symmetric autoencoder:
      Input → hidden layers → bottleneck → mirrored hidden layers → reconstruction
    Returns (autoencoder, encoder) — both as Keras Model objects.
    """
    inp = Input(shape=(input_dim,), name="ae_input")
    x   = inp

    # ── Encoder ──────────────────────────────────────────────
    for i, units in enumerate(hidden_dims):
        x = Dense(units, activation="relu", name=f"enc_dense_{i}")(x)
        x = BatchNormalization(name=f"enc_bn_{i}")(x)
        x = Dropout(dropout, name=f"enc_drop_{i}")(x)

    bottleneck = Dense(encoding_dim, activation="relu", name="bottleneck")(x)

    # ── Decoder ──────────────────────────────────────────────
    x = bottleneck
    for i, units in enumerate(reversed(hidden_dims)):
        x = Dense(units, activation="relu", name=f"dec_dense_{i}")(x)
        x = BatchNormalization(name=f"dec_bn_{i}")(x)
        x = Dropout(dropout, name=f"dec_drop_{i}")(x)

    reconstruction = Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    autoencoder = Model(inp, reconstruction, name="autoencoder")
    encoder     = Model(inp, bottleneck,     name="encoder")

    autoencoder.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    return autoencoder, encoder


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────
def plot_ae_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "autoencoder_loss.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved AE loss plot → {path}")


def plot_comparison(baseline_metrics, encoded_metrics):
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    baseline_vals = [
        baseline_metrics["accuracy"],
        baseline_metrics["precision"],
        baseline_metrics["recall"],
        baseline_metrics["f1_score"],
    ]
    encoded_vals = [
        encoded_metrics["accuracy"],
        encoded_metrics["precision"],
        encoded_metrics["recall"],
        encoded_metrics["f1_score"],
    ]

    x     = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline TF-IDF + LR", color="#3b82f6")
    ax.bar(x + width / 2, encoded_vals,  width, label="Autoencoder + LR",      color="#8b5cf6")

    ax.set_ylabel("Score")
    ax.set_title("Unsupervised Feature Extraction — Baseline vs Autoencoder")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "unsupervised_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved comparison plot → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()

    # ── Train / test split ───────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"],
        df["generated"],
        test_size=0.2,
        random_state=42,
        stratify=df["generated"],
    )
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    # ── TF-IDF vectorisation ─────────────────────────────────
    print("\n[1/5] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf  = vectorizer.transform(X_test_raw)

    # Convert to dense (required for Keras)
    X_train_dense = X_train_tfidf.toarray().astype(np.float32)
    X_test_dense  = X_test_tfidf.toarray().astype(np.float32)

    # Scale to [0, 1] — important for sigmoid reconstruction
    scaler        = MaxAbsScaler()
    X_train_dense = scaler.fit_transform(X_train_dense)
    X_test_dense  = scaler.transform(X_test_dense)

    # ── Baseline: LR on raw TF-IDF ───────────────────────────
    print("\n[2/5] Training baseline classifier (TF-IDF → Logistic Regression)...")
    baseline_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    baseline_clf.fit(X_train_tfidf, y_train)
    y_pred_baseline = baseline_clf.predict(X_test_tfidf)

    baseline_metrics = evaluate_classification(
        y_test, y_pred_baseline, model_name="Baseline TF-IDF + LR"
    )
    print_evaluation(baseline_metrics)

    # ── Train Autoencoder (unsupervised — labels NOT used) ───
    print("\n[3/5] Building and training Autoencoder (unsupervised)...")
    input_dim = X_train_dense.shape[1]
    autoencoder, encoder = build_autoencoder(
        input_dim   = input_dim,
        hidden_dims = HIDDEN_DIMS,
        encoding_dim= ENCODING_DIM,
        dropout     = DROPOUT_RATE,
    )
    autoencoder.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = autoencoder.fit(
        X_train_dense, X_train_dense,   # ← unsupervised: input = target
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    plot_ae_loss(history)

    # ── Extract encoded representations ──────────────────────
    print("\n[4/5] Extracting encoded features...")
    X_train_encoded = encoder.predict(X_train_dense, verbose=0)
    X_test_encoded  = encoder.predict(X_test_dense,  verbose=0)
    print(f"Encoded feature shape: {X_train_encoded.shape}")

    # ── Classifier on encoded features ───────────────────────
    print("\n[5/5] Training classifier on autoencoder features...")
    encoded_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    encoded_clf.fit(X_train_encoded, y_train)
    y_pred_encoded = encoded_clf.predict(X_test_encoded)

    encoded_metrics = evaluate_classification(
        y_test, y_pred_encoded, model_name="Autoencoder Encoded + LR"
    )
    print_evaluation(encoded_metrics)

    # ── Comparison plot ──────────────────────────────────────
    plot_comparison(baseline_metrics, encoded_metrics)

    # ── Save artefacts ───────────────────────────────────────
    encoder.save(ENCODER_PATH)
    print(f"Saved encoder model → {ENCODER_PATH}")

    with open(AE_VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(AE_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(AE_CLF_PATH, "wb") as f:
        pickle.dump(encoded_clf, f)

    config = {
        "tfidf_max_features" : TFIDF_MAX_FEATURES,
        "encoding_dim"       : ENCODING_DIM,
        "hidden_dims"        : HIDDEN_DIMS,
        "dropout_rate"       : DROPOUT_RATE,
        "epochs"             : EPOCHS,
        "batch_size"         : BATCH_SIZE,
        "baseline": {
            "accuracy" : baseline_metrics["accuracy"],
            "precision": baseline_metrics["precision"],
            "recall"   : baseline_metrics["recall"],
            "f1_score" : baseline_metrics["f1_score"],
        },
        "autoencoder_encoded": {
            "accuracy" : encoded_metrics["accuracy"],
            "precision": encoded_metrics["precision"],
            "recall"   : encoded_metrics["recall"],
            "f1_score" : encoded_metrics["f1_score"],
        },
    }
    with open(AE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*55)
    print("UNSUPERVISED EXPERIMENT — FINAL SUMMARY")
    print("="*55)
    print(f"{'Model':<35} {'Acc':>7} {'F1':>7}")
    print("-"*55)
    for m in [baseline_metrics, encoded_metrics]:
        print(f"{m['model_name']:<35} {m['accuracy']:>7.4f} {m['f1_score']:>7.4f}")
    print("="*55)

    delta_f1 = encoded_metrics["f1_score"] - baseline_metrics["f1_score"]
    print(f"\nF1 improvement from autoencoder features: {delta_f1:+.4f}")
    print(f"Saved encoder      → {ENCODER_PATH}")
    print(f"Saved vectorizer   → {AE_VEC_PATH}")
    print(f"Saved classifier   → {AE_CLF_PATH}")
    print(f"Saved config       → {AE_CONFIG_PATH}")


if __name__ == "__main__":
    main()