"""
backend/train_ensemble.py

Loads all trained models, averages their probability outputs, and reports:
  - Individual model metrics
  - Simple average ensemble metrics
  - F1-weighted ensemble metrics
  - Combined ROC curves
  - Overall model comparison bar chart
  - Summary JSON saved to results/

Run AFTER all four base models have been trained.
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.evaluation import (
    evaluate_classification,
    print_evaluation,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_combined_roc_curves,
    plot_model_comparison_bar,
)


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


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
# Model probability collectors
# ─────────────────────────────────────────────────────────────
def get_traditional_proba(X_test_raw):
    model_path = os.path.join(MODEL_DIR, "traditional_model.pkl")
    vec_path   = os.path.join(MODEL_DIR, "traditional_vectorizer.pkl")
    if not os.path.exists(model_path):
        print("[SKIP] traditional model not found.")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    X = vectorizer.transform([clean_text(t) for t in X_test_raw])
    return model.predict_proba(X)[:, 1]


def get_lstm_proba(X_test_raw):
    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    tok_path   = os.path.join(MODEL_DIR, "lstm_tokenizer.pkl")
    cfg_path   = os.path.join(MODEL_DIR, "lstm_config.json")
    if not os.path.exists(model_path):
        print("[SKIP] LSTM model not found.")
        return None
    model = tf.keras.models.load_model(model_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(cfg_path) as f:
        cfg = json.load(f)
    seqs   = tokenizer.texts_to_sequences([clean_text(t) for t in X_test_raw])
    padded = pad_sequences(seqs, maxlen=cfg["max_len"], padding="post", truncating="post")
    return model.predict(padded, verbose=0).flatten()


def get_gru_proba(X_test_raw):
    model_path = os.path.join(MODEL_DIR, "gru_model.keras")
    tok_path   = os.path.join(MODEL_DIR, "gru_tokenizer.pkl")
    cfg_path   = os.path.join(MODEL_DIR, "gru_config.json")
    if not os.path.exists(model_path):
        print("[SKIP] GRU model not found.")
        return None
    model = tf.keras.models.load_model(model_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(cfg_path) as f:
        cfg = json.load(f)
    seqs   = tokenizer.texts_to_sequences([clean_text(t) for t in X_test_raw])
    padded = pad_sequences(seqs, maxlen=cfg["max_len"], padding="post", truncating="post")
    return model.predict(padded, verbose=0).flatten()


def get_bert_proba(X_test_raw):
    bert_dir = os.path.join(MODEL_DIR, "bert_model")
    if not os.path.exists(bert_dir):
        print("[SKIP] BERT model not found.")
        return None
    import torch
    from transformers import BertTokenizerFast, BertForSequenceClassification

    tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
    model     = BertForSequenceClassification.from_pretrained(bert_dir)
    model.eval()

    probs_all = []
    batch_size = 16
    texts = list(X_test_raw)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out    = model(**inputs)
            p      = torch.softmax(out.logits, dim=1)[:, 1].numpy()
            probs_all.extend(p.tolist())

    return np.array(probs_all)


# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()

    _, X_test_raw, _, y_test = train_test_split(
        df["text"], df["generated"],
        test_size=0.2, random_state=42, stratify=df["generated"],
    )
    y_test = np.array(y_test)

    # ── Collect probabilities from each model ────────────────
    print("\nCollecting predictions from all models...")
    collectors = {
        "Logistic Regression": get_traditional_proba,
        "LSTM"               : get_lstm_proba,
        "GRU"                : get_gru_proba,
        "BERT"               : get_bert_proba,
    }

    model_probas = {}
    all_results  = []

    for name, fn in collectors.items():
        print(f"\n── {name} ──")
        proba = fn(X_test_raw)
        if proba is None:
            continue
        model_probas[name] = proba
        pred    = (proba >= 0.5).astype(int)
        metrics = evaluate_classification(y_test, pred, model_name=name, y_proba=proba)
        print_evaluation(metrics)
        plot_confusion_matrix(y_test, pred, name, RESULTS_DIR)
        plot_roc_curve(y_test, proba, name, RESULTS_DIR)
        all_results.append(metrics)

    if len(model_probas) < 2:
        print("\nNeed at least 2 trained models to build an ensemble. Exiting.")
        return

    # ── Simple average ensemble ──────────────────────────────
    print("\n── Simple Average Ensemble ──")
    avg_proba   = np.mean(list(model_probas.values()), axis=0)
    avg_pred    = (avg_proba >= 0.5).astype(int)
    avg_metrics = evaluate_classification(
        y_test, avg_pred, model_name="Ensemble (avg)", y_proba=avg_proba
    )
    print_evaluation(avg_metrics)
    plot_confusion_matrix(y_test, avg_pred, "Ensemble_avg", RESULTS_DIR)
    plot_roc_curve(y_test, avg_proba, "Ensemble_avg", RESULTS_DIR)
    all_results.append(avg_metrics)
    model_probas["Ensemble (avg)"] = avg_proba

    # ── F1-weighted ensemble ─────────────────────────────────
    print("\n── F1-Weighted Ensemble ──")
    f1_weights = np.array([
        m["f1_score"] for m in all_results
        if m["model_name"] not in ("Ensemble (avg)", "Ensemble (weighted)")
    ])
    f1_weights = f1_weights / f1_weights.sum()

    base_probas  = [v for k, v in model_probas.items() if "Ensemble" not in k]
    weighted_proba = np.average(base_probas, axis=0, weights=f1_weights)
    weighted_pred  = (weighted_proba >= 0.5).astype(int)
    weighted_metrics = evaluate_classification(
        y_test, weighted_pred, model_name="Ensemble (weighted)", y_proba=weighted_proba
    )
    print_evaluation(weighted_metrics)
    plot_confusion_matrix(y_test, weighted_pred, "Ensemble_weighted", RESULTS_DIR)
    plot_roc_curve(y_test, weighted_proba, "Ensemble_weighted", RESULTS_DIR)
    all_results.append(weighted_metrics)

    # ── Combined plots ───────────────────────────────────────
    plot_combined_roc_curves(all_results, RESULTS_DIR)
    plot_model_comparison_bar(all_results, RESULTS_DIR)

    # ── Save summary JSON ────────────────────────────────────
    summary = []
    for r in all_results:
        summary.append({
            "model"    : r["model_name"],
            "accuracy" : round(r["accuracy"],  4),
            "precision": round(r["precision"], 4),
            "recall"   : round(r["recall"],    4),
            "f1_score" : round(r["f1_score"],  4),
            "roc_auc"  : round(r["roc_auc"],   4) if r.get("roc_auc") else None,
        })

    summary_path = os.path.join(RESULTS_DIR, "all_models_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # ── Final table ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ALL MODELS — FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<26} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 70)
    for r in all_results:
        auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "  N/A "
        print(
            f"{r['model_name']:<26} "
            f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} {r['f1_score']:>7.4f} {auc:>7}"
        )
    print("=" * 70)

    print(f"\nAll plots  → {RESULTS_DIR}/")
    print(f"Summary    → {summary_path}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()