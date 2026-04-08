# -*- coding: utf-8 -*-
"""
Word2Vec + Logistic Regression — AI vs Human Text Classification
Full Dataset Version | Spyder Compatible

@author: Mdsaz
"""

# 1. IMPORTS

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
import warnings
import logging
logging.disable(logging.CRITICAL) 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)


# 2. LOAD DATASET

df = pd.read_csv(r"C:\Users\Mdsaz\Downloads\AI_Human.csv")

print("DATASET OVERVIEW")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:", df.shape)


# 3. SELECT RELEVANT COLUMNS

TEXT_COLUMN  = "text"       # input text column
LABEL_COLUMN = "generated"  # target label (0 = Human, 1 = AI)

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()         # drop missing values

df[TEXT_COLUMN]  = df[TEXT_COLUMN].astype(str)         # ensure text is string
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)        # ensure label is integer

print("\nCLASS DISTRIBUTION")
print(df[LABEL_COLUMN].value_counts())


# 4. TEXT CLEANING

def clean_text(text):
    text = text.lower()                                        # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                     # keep letters only
    text = re.sub(r"\s+", " ", text).strip()                   # remove extra spaces
    return text

df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

print("\nSample cleaned text:")
print(df["clean_text"].iloc[0])


# 5. TOKENIZATION

df["tokens"] = df["clean_text"].apply(lambda x: x.split())    # word tokenization

df = df[df["tokens"].apply(len) > 0].reset_index(drop=True)   # remove empty rows

sentences = df["tokens"].tolist()    # list of token lists
labels    = df[LABEL_COLUMN].values  # numpy array of labels

print("\nExample tokens (first 20):")
print(sentences[0][:20])


# 6. TRAIN-TEST SPLIT

X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(
    sentences,
    labels,
    test_size=0.2,      # 80% train / 20% test
    random_state=14,    # reproducibility
    stratify=labels     # preserve class balance
)

print("\nTRAIN-TEST SPLIT")
print("Training samples :", len(X_train_tokens))
print("Testing  samples :", len(X_test_tokens))


# 7. TRAIN WORD2VEC MODEL (Full Training Set)

print("\nTraining Word2Vec on full training data...")

word2vec_model = Word2Vec(
    sentences=X_train_tokens,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=0,
    epochs=1
)

for epoch in tqdm(range(50), desc="Training Epochs", unit="epoch"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        word2vec_model.train(
            X_train_tokens,
            total_examples=word2vec_model.corpus_count,
            epochs=1
        )


print("Word2Vec vocabulary size:", len(word2vec_model.wv))



# 8. DOCUMENT VECTORIZATION (Mean Pooling)

def document_vector(tokens, model):
    """Average Word2Vec vectors for all known tokens in a document."""
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)   # zero vector for unknown docs
    return np.mean(word_vectors, axis=0)     # mean of all word vectors

print("\nVectorizing training data...")
X_train_vec = np.array([document_vector(tokens, word2vec_model)
                         for tokens in X_train_tokens])

print("Vectorizing test data...")
X_test_vec  = np.array([document_vector(tokens, word2vec_model)
                         for tokens in X_test_tokens])

print("\nTrain vector shape:", X_train_vec.shape)
print("Test  vector shape:", X_test_vec.shape)


# 9. LOGISTIC REGRESSION CLASSIFIER

print("\nTraining Logistic Regression classifier...")

classifier = LogisticRegression(max_iter=1000, random_state=14)
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)

print("Training complete.")


# 10. MODEL EVALUATION

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\nEVALUATION RESULTS")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))


# 11. CONFUSION MATRIX VISUALIZATION

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=True)
plt.title("Confusion Matrix — Word2Vec + Logistic Regression", fontsize=13)
plt.tight_layout()
plt.show()