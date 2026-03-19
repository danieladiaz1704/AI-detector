import os
import re
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model", "bert_model")

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
SAMPLE_SIZE = 20000
NUM_EPOCHS = 1


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
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["generated"] = df["generated"].astype(int)

    print(f"Original rows: {len(df)}")

    if len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} rows for faster BERT training...")

        # balanceado manual
        half = SAMPLE_SIZE // 2

        human_df = df[df["generated"] == 0].sample(n=half, random_state=42)
        ai_df = df[df["generated"] == 1].sample(n=half, random_state=42)

        df = pd.concat([human_df, ai_df])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Rows used for BERT: {len(df)}")
    return df

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()

    print("Splitting dataset...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["generated"]
    )

    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    print("Creating Hugging Face datasets...")
    train_dataset = Dataset.from_pandas(train_df.rename(columns={"generated": "labels"}))
    test_dataset = Dataset.from_pandas(test_df.rename(columns={"generated": "labels"}))

    print("Tokenizing training dataset...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    print("Tokenizing test dataset...")
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "bert_results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none"
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Training BERT...")
    trainer.train()

    print("Evaluating BERT...")
    eval_results = trainer.evaluate()

    print("\n===== BERT Evaluation =====")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    print("Saving BERT model...")
    trainer.save_model(MODEL_DIR)

    print("Saving tokenizer...")
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"\nSaved BERT model and tokenizer to: {MODEL_DIR}")
    print("BERT training completed successfully.")


if __name__ == "__main__":
    main()