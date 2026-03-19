import os
import re
import json
import pickle
import joblib
import numpy as np
import torch

from typing import List, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizerFast, BertForSequenceClassification


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

TRADITIONAL_MODEL_PATH = os.path.join(MODEL_DIR, "traditional_model.pkl")
TRADITIONAL_VECTORIZER_PATH = os.path.join(MODEL_DIR, "traditional_vectorizer.pkl")

LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
LSTM_TOKENIZER_PATH = os.path.join(MODEL_DIR, "lstm_tokenizer.pkl")
LSTM_CONFIG_PATH = os.path.join(MODEL_DIR, "lstm_config.json")

GRU_MODEL_PATH = os.path.join(MODEL_DIR, "gru_model.keras")
GRU_TOKENIZER_PATH = os.path.join(MODEL_DIR, "gru_tokenizer.pkl")
GRU_CONFIG_PATH = os.path.join(MODEL_DIR, "gru_config.json")

BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert_model")


app = FastAPI(title="AI Text Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    model_name: Literal["traditional", "lstm", "gru", "bert"] = "traditional"


class SentencePrediction(BaseModel):
    sentence: str
    ai_probability: float
    human_probability: float
    label: str
    highlight_color: str


class PredictResponse(BaseModel):
    model_name: str
    ai_probability: float
    human_probability: float
    label: str
    sentences: List[SentencePrediction]


traditional_model = None
traditional_vectorizer = None

lstm_model = None
lstm_tokenizer = None
lstm_config = None

gru_model = None
gru_tokenizer = None
gru_config = None

bert_model = None
bert_tokenizer = None

bert_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def load_traditional():
    global traditional_model, traditional_vectorizer

    if traditional_model is None or traditional_vectorizer is None:
        traditional_model = joblib.load(TRADITIONAL_MODEL_PATH)
        traditional_vectorizer = joblib.load(TRADITIONAL_VECTORIZER_PATH)


def load_lstm():
    global lstm_model, lstm_tokenizer, lstm_config

    if lstm_model is None:
        lstm_model = load_model(LSTM_MODEL_PATH)

        with open(LSTM_TOKENIZER_PATH, "rb") as f:
            lstm_tokenizer = pickle.load(f)

        with open(LSTM_CONFIG_PATH, "r", encoding="utf-8") as f:
            lstm_config = json.load(f)


def load_gru():
    global gru_model, gru_tokenizer, gru_config

    if gru_model is None:
        gru_model = load_model(GRU_MODEL_PATH)

        with open(GRU_TOKENIZER_PATH, "rb") as f:
            gru_tokenizer = pickle.load(f)

        with open(GRU_CONFIG_PATH, "r", encoding="utf-8") as f:
            gru_config = json.load(f)


def load_bert():
    global bert_model, bert_tokenizer

    if bert_model is None or bert_tokenizer is None:
        bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_PATH)
        bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        bert_model.to(bert_device)
        bert_model.eval()


def traditional_predict_proba(texts: List[str]) -> np.ndarray:
    load_traditional()
    cleaned = [clean_text(t) for t in texts]
    vectors = traditional_vectorizer.transform(cleaned)
    probs = traditional_model.predict_proba(vectors)[:, 1]
    return probs


def lstm_predict_proba(texts: List[str]) -> np.ndarray:
    load_lstm()
    cleaned = [clean_text(t).lower() for t in texts]
    sequences = lstm_tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(
        sequences,
        maxlen=lstm_config["max_len"],
        padding="post",
        truncating="post"
    )
    probs = lstm_model.predict(padded, verbose=0).flatten()
    return probs


def gru_predict_proba(texts: List[str]) -> np.ndarray:
    load_gru()
    cleaned = [clean_text(t).lower() for t in texts]
    sequences = gru_tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(
        sequences,
        maxlen=gru_config["max_len"],
        padding="post",
        truncating="post"
    )
    probs = gru_model.predict(padded, verbose=0).flatten()
    return probs


def bert_predict_proba(texts: List[str]) -> np.ndarray:
    load_bert()
    cleaned = [clean_text(t) for t in texts]

    inputs = bert_tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(bert_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

    return probs


def predict_with_model(model_name: str, texts: List[str]) -> np.ndarray:
    if model_name == "traditional":
        return traditional_predict_proba(texts)
    if model_name == "lstm":
        return lstm_predict_proba(texts)
    if model_name == "gru":
        return gru_predict_proba(texts)
    if model_name == "bert":
        return bert_predict_proba(texts)

    raise ValueError(f"Unsupported model: {model_name}")


def build_sentence_results(sentences: List[str], probs: np.ndarray) -> List[Dict[str, Any]]:
    results = []

    for sentence, ai_prob in zip(sentences, probs):
        ai_prob = float(ai_prob)
        human_prob = float(1 - ai_prob)
        label = "AI" if ai_prob >= 0.5 else "Human"
        color = "red" if label == "AI" else "green"

        results.append({
            "sentence": sentence,
            "ai_probability": round(ai_prob * 100, 2),
            "human_probability": round(human_prob * 100, 2),
            "label": label,
            "highlight_color": color
        })

    return results


@app.get("/")
def root():
    return {"message": "AI Text Detector API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = clean_text(request.text)

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        overall_prob = predict_with_model(request.model_name, [text])[0]

        sentences = split_into_sentences(text)
        if not sentences:
            sentences = [text]

        sentence_probs = predict_with_model(request.model_name, sentences)
        sentence_results = build_sentence_results(sentences, sentence_probs)

        ai_probability = float(overall_prob)
        human_probability = float(1 - overall_prob)
        label = "AI" if ai_probability >= 0.5 else "Human"

        return {
            "model_name": request.model_name,
            "ai_probability": round(ai_probability * 100, 2),
            "human_probability": round(human_probability * 100, 2),
            "label": label,
            "sentences": sentence_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))