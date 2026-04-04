"""
backend/train_bert.py

Experiment 3 — State-of-the-art: BERT fine-tuning.
  - 3 training epochs (increased from 1)
  - Saves accuracy/loss curves from trainer log history
  - Reports ROC-AUC alongside standard metrics
"""

import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
import re
import json
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.evaluation import (
    evaluate_classification,
    print_evaluation,
    plot_confusion_matrix,
    plot_roc_curve,
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model", "bert_model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 128
SAMPLE_SIZE = 20_000
NUM_EPOCHS  = 3          # increased from 1


# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).strip()
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

    print(f"Original rows: {len(df)}")

    if len(df) > SAMPLE_SIZE:
        half     = SAMPLE_SIZE // 2
        human_df = df[df["generated"] == 0].sample(n=half, random_state=42)
        ai_df    = df[df["generated"] == 1].sample(n=half, random_state=42)
        df       = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Rows used for BERT: {len(df)}")
    return df


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs       = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = 0.0

    return {
        "accuracy" : acc,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "roc_auc"  : auc,
    }


# ─────────────────────────────────────────────────────────────
def save_bert_plots(log_history: list, results_dir: str) -> None:
    """Parse Trainer log_history and save loss + accuracy curves."""
    os.makedirs(results_dir, exist_ok=True)

    # Training steps: have 'loss' but NOT 'eval_loss'
    train_steps  = [x["step"] for x in log_history if "loss" in x and "eval_loss" not in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]

    # Eval entries: have 'eval_loss'
    eval_entries = [x for x in log_history if "eval_loss" in x]
    eval_epochs  = [x.get("epoch", i + 1) for i, x in enumerate(eval_entries)]
    eval_losses  = [x["eval_loss"] for x in eval_entries]
    eval_accs    = [x.get("eval_accuracy", None) for x in eval_entries]
    eval_aucs    = [x.get("eval_roc_auc", None) for x in eval_entries]

    # ── Loss plot ─────────────────────────────────────────────
    if train_losses:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_steps, train_losses, label="Train Loss", lw=1.5)
        if eval_losses:
            # Normalise eval x-axis to match step range
            ax.plot(
                np.linspace(train_steps[0], train_steps[-1], len(eval_losses)),
                eval_losses, label="Val Loss", lw=1.5, linestyle="--",
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("BERT — Training / Validation Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, "BERT_loss.png"), dpi=150)
        plt.close(fig)
        print(f"Saved BERT loss plot → {results_dir}/BERT_loss.png")

    # ── Accuracy plot ─────────────────────────────────────────
    valid_accs = [(e, a) for e, a in zip(eval_epochs, eval_accs) if a is not None]
    if valid_accs:
        epochs, accs = zip(*valid_accs)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, accs, marker="o", lw=2, label="Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("BERT — Validation Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, "BERT_accuracy.png"), dpi=150)
        plt.close(fig)
        print(f"Saved BERT accuracy plot → {results_dir}/BERT_accuracy.png")

    # ── AUC plot ─────────────────────────────────────────────
    valid_aucs = [(e, a) for e, a in zip(eval_epochs, eval_aucs) if a is not None]
    if valid_aucs:
        epochs, aucs = zip(*valid_aucs)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, aucs, marker="s", lw=2, color="purple", label="Val ROC-AUC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0, 1.05)
        ax.set_title("BERT — Validation ROC-AUC")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, "BERT_roc_auc.png"), dpi=150)
        plt.close(fig)
        print(f"Saved BERT AUC plot → {results_dir}/BERT_roc_auc.png")


# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()

    print("Splitting dataset...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["generated"]
    )

    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    print("Creating HuggingFace datasets...")
    train_dataset = Dataset.from_pandas(train_df.rename(columns={"generated": "labels"}))
    test_dataset  = Dataset.from_pandas(test_df.rename(columns={"generated": "labels"}))

    print("Tokenizing...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset  = test_dataset.map(lambda x: tokenize_function(x, tokenizer),  batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format( type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir                = os.path.join(BASE_DIR, "bert_checkpoints"),
        eval_strategy             = "epoch",
        save_strategy             = "epoch",
        learning_rate             = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 16,
        num_train_epochs          = NUM_EPOCHS,
        weight_decay              = 0.01,
        warmup_ratio              = 0.1,
        logging_steps             = 50,
        load_best_model_at_end    = True,
        metric_for_best_model     = "f1",
        report_to                 = "none",
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = test_dataset,
        compute_metrics = compute_metrics,
    )

    print("Training BERT (3 epochs)...")
    trainer.train()

    # ── Save training plots ──────────────────────────────────
    save_bert_plots(trainer.state.log_history, RESULTS_DIR)

    # ── Final evaluation (from last epoch log) ───────────────
    print("\n===== BERT Evaluation =====")
    eval_entries = [x for x in trainer.state.log_history if "eval_loss" in x]
    if eval_entries:
        last_eval = eval_entries[-1]
        for key, value in last_eval.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ── Get predictions for confusion matrix + ROC ───────────
    import torch
    pred_output = trainer.predict(test_dataset)
    logits      = pred_output.predictions
    labels      = pred_output.label_ids
    probs       = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds       = (probs >= 0.5).astype(int)

    metrics = evaluate_classification(
        labels, preds, model_name="BERT", y_proba=probs
    )
    print_evaluation(metrics)

    plot_confusion_matrix(labels, preds, "BERT", RESULTS_DIR)
    plot_roc_curve(labels, probs, "BERT", RESULTS_DIR)

    # ── Save model + tokenizer ───────────────────────────────
    print("Saving BERT model...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # ── Save config ──────────────────────────────────────────
    config = {
        "model_name" : MODEL_NAME,
        "max_len"    : MAX_LEN,
        "num_epochs" : NUM_EPOCHS,
        "sample_size": SAMPLE_SIZE,
        "accuracy"   : metrics["accuracy"],
        "precision"  : metrics["precision"],
        "recall"     : metrics["recall"],
        "f1_score"   : metrics["f1_score"],
        "roc_auc"    : metrics["roc_auc"],
    }
    config_path = os.path.join(os.path.dirname(MODEL_DIR), "bert_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\nSaved model  → {MODEL_DIR}")
    print(f"Saved config → {config_path}")
    print(f"Saved plots  → {RESULTS_DIR}/")
    print("BERT training complete.")


if __name__ == "__main__":
    main()