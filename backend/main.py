from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="AI Text Detector API",
    description="API for detecting whether a text is human-written or AI-generated.",
    version="1.0.0"
)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and vectorizer
model = joblib.load("model/text_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


class TextInput(BaseModel):
    text: str = Field(..., min_length=5, description="Input text to analyze")


@app.get("/")
def home():
    return {
        "message": "AI Text Detector API is running",
        "status": "online"
    }


@app.post("/predict")
def predict(data: TextInput):
    text = data.text.strip()

    # Validate input
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Convert text into vector format
    text_vectorized = vectorizer.transform([text])

    # Get probabilities for both classes
    probabilities = model.predict_proba(text_vectorized)[0]

    # Class 0 = Human-written, Class 1 = AI-generated
    human_probability = float(probabilities[0]) * 100
    ai_probability = float(probabilities[1]) * 100

    return {
        "ai_percentage": round(ai_probability, 2),
        "human_percentage": round(human_probability, 2)
    }