"""
backend/app.py
Production-hardened FastAPI server.
  - Structured logging with timestamps
  - In-memory rate limiting (30 req/min per IP)
  - Input validation (max 5000 chars, non-empty)
  - Lazy model loading with status endpoint
  - Supports: traditional, lstm, gru, bert, ensemble
  - Sentence-level highlighting for all models

Run: uvicorn app:app --host 127.0.0.1 --port 8001 --reload
"""

import os
import re
import json
import pickle
import logging
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import nltk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai-detector")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AI Text Detector API",
    description = "Detects AI-generated text using LSTM, GRU, BERT, and ensemble models.",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

VALID_MODELS    = {"traditional", "lstm", "gru", "bert", "ensemble"}
MAX_TEXT_LENGTH = 5_000   # characters
MIN_TEXT_LENGTH = 10
RATE_LIMIT      = 30      # requests per minute per IP
BERT_SENT_LIMIT = 10

_cache: dict = {}


# ─────────────────────────────────────────────────────────────
# Rate limiter (in-memory, per IP)
# ─────────────────────────────────────────────────────────────
_rate_store: dict = defaultdict(list)

def check_rate_limit(ip: str) -> bool:
    """Returns True if the request is allowed, False if over limit."""
    now     = time.time()
    window  = 60.0
    history = _rate_store[ip]
    history[:] = [t for t in history if now - t < window]
    if len(history) >= RATE_LIMIT:
        return False
    history.append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/predict":
        client_ip = request.client.host if request.client else "unknown"
        if not check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded | IP={client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait 60 seconds."},
            )
    return await call_next(request)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _require_file(*paths: str) -> None:
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")


# ─────────────────────────────────────────────────────────────
# Model loaders (lazy)
# ─────────────────────────────────────────────────────────────
def load_traditional():
    if "traditional" not in _cache:
        mp = os.path.join(MODEL_DIR, "traditional_model.pkl")
        vp = os.path.join(MODEL_DIR, "traditional_vectorizer.pkl")
        _require_file(mp, vp)
        with open(mp, "rb") as f: model = pickle.load(f)
        with open(vp, "rb") as f: vec   = pickle.load(f)
        _cache["traditional"] = (model, vec)
        logger.info("Traditional model loaded.")
    return _cache["traditional"]


def load_lstm():
    if "lstm" not in _cache:
        import tensorflow as tf
        mp  = os.path.join(MODEL_DIR, "lstm_model.keras")
        tp  = os.path.join(MODEL_DIR, "lstm_tokenizer.pkl")
        cp  = os.path.join(MODEL_DIR, "lstm_config.json")
        _require_file(mp, tp, cp)
        model = tf.keras.models.load_model(mp)
        with open(tp, "rb") as f: tok = pickle.load(f)
        with open(cp)        as f: cfg = json.load(f)
        _cache["lstm"] = (model, tok, cfg)
        logger.info("LSTM model loaded.")
    return _cache["lstm"]


def load_gru():
    if "gru" not in _cache:
        import tensorflow as tf
        mp  = os.path.join(MODEL_DIR, "gru_model.keras")
        tp  = os.path.join(MODEL_DIR, "gru_tokenizer.pkl")
        cp  = os.path.join(MODEL_DIR, "gru_config.json")
        _require_file(mp, tp, cp)
        model = tf.keras.models.load_model(mp)
        with open(tp, "rb") as f: tok = pickle.load(f)
        with open(cp)        as f: cfg = json.load(f)
        _cache["gru"] = (model, tok, cfg)
        logger.info("GRU model loaded.")
    return _cache["gru"]


def load_bert():
    if "bert" not in _cache:
        import torch
        from transformers import BertTokenizerFast, BertForSequenceClassification
        bert_dir = os.path.join(MODEL_DIR, "bert_model")
        if not os.path.isdir(bert_dir):
            raise FileNotFoundError(f"BERT model directory not found: {bert_dir}")
        tok   = BertTokenizerFast.from_pretrained(bert_dir)
        model = BertForSequenceClassification.from_pretrained(bert_dir)
        model.eval()
        _cache["bert"] = (model, tok)
        logger.info("BERT model loaded.")
    return _cache["bert"]


# ─────────────────────────────────────────────────────────────
# Prediction functions
# ─────────────────────────────────────────────────────────────
def predict_traditional(texts: list) -> np.ndarray:
    model, vec = load_traditional()
    X = vec.transform([clean_text(t) for t in texts])
    return model.predict_proba(X)[:, 1]


def predict_lstm(texts: list) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model, tok, cfg = load_lstm()
    seqs   = tok.texts_to_sequences([clean_text(t) for t in texts])
    padded = pad_sequences(seqs, maxlen=cfg["max_len"], padding="post", truncating="post")
    return model.predict(padded, verbose=0).flatten()


def predict_gru(texts: list) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model, tok, cfg = load_gru()
    seqs   = tok.texts_to_sequences([clean_text(t) for t in texts])
    padded = pad_sequences(seqs, maxlen=cfg["max_len"], padding="post", truncating="post")
    return model.predict(padded, verbose=0).flatten()


def predict_bert(texts: list) -> np.ndarray:
    import torch
    model, tok = load_bert()
    inputs = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out   = model(**inputs)
        probs = torch.softmax(out.logits, dim=1)[:, 1].numpy()
    return probs


def predict_ensemble(texts: list) -> np.ndarray:
    """Average all available model probabilities."""
    fns = [predict_traditional, predict_lstm, predict_gru, predict_bert]
    all_probs = []
    for fn in fns:
        try:
            all_probs.append(fn(texts))
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Ensemble: skipping {fn.__name__}: {e}")
    if not all_probs:
        raise RuntimeError("No models available for ensemble.")
    return np.mean(all_probs, axis=0)


PREDICT_FUNCS = {
    "traditional": predict_traditional,
    "lstm"        : predict_lstm,
    "gru"         : predict_gru,
    "bert"        : predict_bert,
    "ensemble"    : predict_ensemble,
}


# ─────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_name: str = Field(default="lstm")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if len(v) < MIN_TEXT_LENGTH:
            raise ValueError(f"Text must be at least {MIN_TEXT_LENGTH} characters.")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text is too long ({len(v)} chars). "
                f"Maximum allowed is {MAX_TEXT_LENGTH} characters."
            )
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_MODELS:
            raise ValueError(
                f"Unknown model '{v}'. Choose from: {sorted(VALID_MODELS)}"
            )
        return v


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    loaded = list(_cache.keys())
    return {"status": "ok", "models_loaded": loaded}


@app.get("/models")
def list_models():
    """Return available model names and their availability."""
    status = {}
    checks = {
        "traditional": [
            os.path.join(MODEL_DIR, "traditional_model.pkl"),
        ],
        "lstm" : [os.path.join(MODEL_DIR, "lstm_model.keras")],
        "gru"  : [os.path.join(MODEL_DIR, "gru_model.keras")],
        "bert" : [os.path.join(MODEL_DIR, "bert_model")],
    }
    for name, paths in checks.items():
        status[name] = all(os.path.exists(p) for p in paths)
    status["ensemble"] = sum(status.values()) >= 2
    return {"models": status}


@app.post("/predict")
def predict(request: PredictRequest, req: Request):
    model_name = request.model_name
    client_ip  = req.client.host if req.client else "unknown"
    start      = time.time()

    logger.info(f"Predict request | model={model_name} | ip={client_ip} | chars={len(request.text)}")

    try:
        predict_fn = PREDICT_FUNCS[model_name]

        # ── Overall score ───────────────────────────────────
        overall_ai_prob = float(predict_fn([request.text])[0])

        # ── Sentence-level ──────────────────────────────────
        raw_sents = sent_tokenize(request.text)
        sents     = [s for s in raw_sents if len(s.split()) >= 4]

        sentence_results = []
        if sents:
            cap    = BERT_SENT_LIMIT if model_name in ("bert", "ensemble") else len(sents)
            to_run = sents[:cap]

            sent_probs = predict_fn(to_run)

            for sent, prob in zip(to_run, sent_probs):
                sentence_results.append({
                    "sentence"   : sent,
                    "label"      : "AI" if float(prob) >= 0.5 else "Human",
                    "probability": round(float(prob), 4),
                })

            for sent in sents[cap:]:
                sentence_results.append({
                    "sentence"   : sent,
                    "label"      : "AI" if overall_ai_prob >= 0.5 else "Human",
                    "probability": round(overall_ai_prob, 4),
                })

        elapsed = round(time.time() - start, 3)
        logger.info(
            f"Predict done | model={model_name} | "
            f"ai_prob={overall_ai_prob:.3f} | elapsed={elapsed}s"
        )

        return {
            "ai_probability"   : round(overall_ai_prob * 100, 1),
            "human_probability": round((1 - overall_ai_prob) * 100, 1),
            "model_used"       : model_name,
            "sentences"        : sentence_results,
            "processing_time_s": elapsed,
        }

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=503,
            detail=str(e) + " — run the appropriate train script first.",
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")