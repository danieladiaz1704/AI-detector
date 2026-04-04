"""
backend/train_traditional.py

Experiment 1 (supplementary) — Traditional ML baseline:
  TF-IDF vectorization + Logistic Regression.
  Runs 3 experiments varying regularization strength (C) and n-gram range.
  Saves the best model for use by the API.
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.evaluation import evaluate_classification, print_evaluation


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

MODEL_PATH      = os.path.join(MODEL_DIR, "traditional_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "traditional_vectorizer.pkl")
CONFIG_PATH     = os.path.join(MODEL_DIR, "traditional_config.json")


# ─────────────────────────────────────────────────────────────
# Helpers
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

    # Sample to avoid MemoryError on large datasets
    SAMPLE_SIZE = 50_000
    if len(df) > SAMPLE_SIZE:
        half     = SAMPLE_SIZE // 2
        human_df = df[df["generated"] == 0].sample(n=half, random_state=42)
        ai_df    = df[df["generated"] == 1].sample(n=half, random_state=42)
        df       = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Rows used: {len(df)}")
    return df


def save_comparison_plot(results: list, plots_dir: str):
    """Bar chart comparing F1-score across all experiments."""
    names  = [r["model_name"] for r in results]
    f1s    = [r["f1_score"]   for r in results]
    accs   = [r["accuracy"]   for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#3b82f6")
    ax.bar(x + width / 2, f1s,  width, label="F1-Score",  color="#8b5cf6")

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("Traditional Model — Experiment Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "traditional_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved comparison plot → {path}")


# ─────────────────────────────────────────────────────────────
# Experiments configuration
# ─────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name"        : "LR_C1_unigram",
        "C"           : 1.0,
        "ngram_range" : (1, 1),
        "max_features": 20_000,   # was 50_000
        "solver"      : "lbfgs",
        "description" : "C=1.0, unigrams (baseline)"
    },
    {
        "name"        : "LR_C01_unigram",
        "C"           : 0.1,
        "ngram_range" : (1, 1),
        "max_features": 20_000,   # was 50_000
        "solver"      : "lbfgs",
        "description" : "C=0.1, unigrams (stronger L2 regularization)"
    },
    {
        "name"        : "LR_C1_bigram",
        "C"           : 1.0,
        "ngram_range" : (1, 2),
        "max_features": 30_000,   # was 80_000
        "solver"      : "lbfgs",
        "description" : "C=1.0, unigrams + bigrams"
    },
]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"],
        df["generated"],
        test_size=0.2,
        random_state=42,
        stratify=df["generated"],
    )

    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    results      = []
    best_f1      = -1.0
    best_model   = None
    best_vec     = None
    best_exp     = None

    for exp in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"Running: {exp['name']}  ({exp['description']})")
        print(f"{'='*50}")

        # ── Vectorise ───────────────────────────────────────────
        vectorizer = TfidfVectorizer(
            ngram_range  = exp["ngram_range"],
            max_features = exp["max_features"],
            sublinear_tf = True,       # log-scale TF
        )
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test  = vectorizer.transform(X_test_raw)

        # ── Train ───────────────────────────────────────────────
        clf = LogisticRegression(
            C          = exp["C"],
            solver     = exp["solver"],
            max_iter   = 1000,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # ── Evaluate ────────────────────────────────────────────
        y_pred   = clf.predict(X_test)
        metrics  = evaluate_classification(y_test, y_pred, model_name=exp["name"])
        print_evaluation(metrics)

        row = {
            "model_name"  : exp["name"],
            "C"           : exp["C"],
            "ngram_range" : str(exp["ngram_range"]),
            "max_features": exp["max_features"],
            "accuracy"    : metrics["accuracy"],
            "precision"   : metrics["precision"],
            "recall"      : metrics["recall"],
            "f1_score"    : metrics["f1_score"],
            "confusion_matrix"      : metrics["confusion_matrix"],
            "classification_report" : metrics["classification_report"],
        }
        results.append(row)

        if metrics["f1_score"] > best_f1:
            best_f1    = metrics["f1_score"]
            best_model = clf
            best_vec   = vectorizer
            best_exp   = exp

    # ── Save best model ─────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(best_vec, f)

    config = {
        "best_experiment" : best_exp["name"],
        "C"               : best_exp["C"],
        "ngram_range"     : str(best_exp["ngram_range"]),
        "max_features"    : best_exp["max_features"],
        "best_f1"         : best_f1,
        "all_results"     : [
            {k: v for k, v in r.items() if k not in ("confusion_matrix", "classification_report")}
            for r in results
        ],
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # ── Comparison plot ─────────────────────────────────────────
    save_comparison_plot(results, PLOTS_DIR)

    # ── Summary ─────────────────────────────────────────────────
    print("\n===== FINAL COMPARISON =====")
    for r in results:
        print(
            f"{r['model_name']:25s} | "
            f"Acc={r['accuracy']:.4f} | "
            f"P={r['precision']:.4f} | "
            f"R={r['recall']:.4f} | "
            f"F1={r['f1_score']:.4f}"
        )

    print(f"\nBest experiment : {best_exp['name']}  (F1={best_f1:.4f})")
    print(f"Saved model     → {MODEL_PATH}")
    print(f"Saved vectorizer→ {VECTORIZER_PATH}")
    print(f"Saved config    → {CONFIG_PATH}")


if __name__ == "__main__":
    main()