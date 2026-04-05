import os
import re
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, GlobalAveragePooling1D,
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# ── Config (matched to LSTM settings) ────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_WORDS    = 20000   # same as LSTM
MAX_LEN      = 200     # same as LSTM
EMBED_DIM    = 128     # same as LSTM (EMBEDDING_DIM)
ENCODING_DIM = 32      # autoencoder bottleneck size
BATCH_SIZE   = 32      # same as LSTM
AE_EPOCHS    = 5       # same as LSTM
CLF_EPOCHS   = 5       # same as LSTM


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    df["text"]      = df["text"].astype(str).apply(clean_text)
    df["generated"] = df["generated"].astype(int)

    print(f"Rows loaded: {len(df)}")
    return df


def save_plot(history, name, metric="loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history[metric],          label=f"train_{metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(f"{name} - {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    path = os.path.join(PLOTS_DIR, f"autoencoder_{name}_{metric}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    df = load_data()

    # ── 1. Tokenise & pad (same pipeline as LSTM) ────────────────────────────
    print("\nTokenising text...")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["generated"],
        test_size=0.2,
        random_state=42,
        stratify=df["generated"]
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq  = tokenizer.texts_to_sequences(X_test_text)

    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="post")

    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    # ── 2. Build Autoencoder ─────────────────────────────────────────────────
    #
    # UNSUPERVISED: learns to compress and reconstruct text embeddings
    # without using any labels during training.
    #
    # Architecture:
    #   Embedding(128) -> GlobalAveragePooling
    #   -> Encoder: Dense(128) -> Dense(64) -> Bottleneck(32)
    #   -> Decoder: Dense(64) -> Dense(128) -> Output(EMBED_DIM)
    #
    print("\n===== Building Autoencoder (Unsupervised) =====")

    inp  = Input(shape=(MAX_LEN,), name="input")
    emb  = Embedding(vocab_size, EMBED_DIM, name="embedding")(inp)
    pool = GlobalAveragePooling1D(name="pooling")(emb)

    # Encoder
    enc1    = Dense(128, activation="relu", name="enc1")(pool)
    enc1    = BatchNormalization()(enc1)
    enc1    = Dropout(0.3)(enc1)
    enc2    = Dense(64,  activation="relu", name="enc2")(enc1)
    encoded = Dense(ENCODING_DIM, activation="relu", name="bottleneck")(enc2)

    # Decoder
    dec1    = Dense(64,        activation="relu",    name="dec1")(encoded)
    dec1    = Dropout(0.3)(dec1)
    dec2    = Dense(128,       activation="relu",    name="dec2")(dec1)
    decoded = Dense(EMBED_DIM, activation="linear",  name="output")(dec2)

    autoencoder = Model(inp, decoded, name="autoencoder")
    encoder     = Model(inp, encoded, name="encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.summary()

    # ── 3. Train Autoencoder (no labels used) ───────────────────────────────
    print("\nTraining autoencoder (unsupervised - labels not used)...")

    # Initialise weights with a dummy pass, then compute reconstruction targets
    autoencoder.fit(X_train[:1], np.zeros((1, EMBED_DIM)), epochs=1, verbose=0)

    emb_weights = autoencoder.get_layer("embedding").get_weights()[0]

    def get_pooled(seqs):
        result = []
        for seq in seqs:
            vecs = emb_weights[seq]
            result.append(vecs.mean(axis=0))
        return np.array(result)

    train_targets = get_pooled(X_train)
    test_targets  = get_pooled(X_test)

    es_ae = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    ae_history = autoencoder.fit(
        X_train, train_targets,
        validation_data=(X_test, test_targets),
        epochs=AE_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es_ae],
        verbose=1
    )

    save_plot(ae_history, "autoencoder", metric="loss")

    # ── 4. Extract features using Encoder ───────────────────────────────────
    print("\nExtracting features using trained encoder...")
    X_train_encoded = encoder.predict(X_train, batch_size=BATCH_SIZE, verbose=1)
    X_test_encoded  = encoder.predict(X_test,  batch_size=BATCH_SIZE, verbose=1)
    print(f"Encoded feature shape: {X_train_encoded.shape}")

    # ── 5. Train Classifier on extracted features ────────────────────────────
    print("\n===== Training Classifier on Autoencoder Features =====")

    clf_inp = Input(shape=(ENCODING_DIM,), name="features")
    x       = Dense(64,  activation="relu")(clf_inp)
    x       = BatchNormalization()(x)
    x       = Dropout(0.3)(x)
    x       = Dense(32,  activation="relu")(x)
    x       = Dropout(0.2)(x)
    out     = Dense(1,   activation="sigmoid")(x)

    classifier = Model(clf_inp, out, name="classifier")
    classifier.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    classifier.summary()

    es_clf = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    clf_history = classifier.fit(
        X_train_encoded, y_train,
        validation_data=(X_test_encoded, y_test),
        epochs=CLF_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es_clf],
        verbose=1
    )

    save_plot(clf_history, "classifier", metric="loss")
    save_plot(clf_history, "classifier", metric="accuracy")

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    print("\n===== Autoencoder + Classifier Evaluation =====")

    y_pred_proba = classifier.predict(X_test_encoded, batch_size=BATCH_SIZE).flatten()
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ── 7. Save everything ───────────────────────────────────────────────────
    ae_path  = os.path.join(MODEL_DIR, "autoencoder_model.keras")
    enc_path = os.path.join(MODEL_DIR, "encoder_model.keras")
    clf_path = os.path.join(MODEL_DIR, "autoencoder_classifier.keras")
    tok_path = os.path.join(MODEL_DIR, "autoencoder_tokenizer.pkl")
    cfg_path = os.path.join(MODEL_DIR, "autoencoder_config.json")
    res_path = os.path.join(RESULTS_DIR, "autoencoder_results.json")

    autoencoder.save(ae_path)
    encoder.save(enc_path)
    classifier.save(clf_path)

    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_words":    MAX_WORDS,
        "max_len":      MAX_LEN,
        "embed_dim":    EMBED_DIM,
        "encoding_dim": ENCODING_DIM,
        "batch_size":   BATCH_SIZE,
        "ae_epochs":    AE_EPOCHS,
        "clf_epochs":   CLF_EPOCHS,
        "threshold":    0.5
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=4)

    results = {
        "experiment":       "Unsupervised Feature Extraction (Autoencoder)",
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm.tolist(),
    }
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved autoencoder  -> {ae_path}")
    print(f"Saved encoder      -> {enc_path}")
    print(f"Saved classifier   -> {clf_path}")
    print(f"Saved tokenizer    -> {tok_path}")
    print(f"Saved config       -> {cfg_path}")
    print(f"Saved results      -> {res_path}")
    print("\nAutoencoder training finished!")


if __name__ == "__main__":
    main()