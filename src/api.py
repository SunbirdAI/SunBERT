"""
API endpoints for text category prediction
"""

from typing import Dict, List

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()

class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    probabilities: Dict[str, float]
    classification: str
    confidence: float

class ClassificationRequestBatch(BaseModel):
    text_list: List[str]

@app.post("/predict", response_model=ClassificationResponse)
def predict(request: ClassificationRequest, model: Model = Depends(get_model)):
    classification, confidence, probabilities = model.predict(request.text)

    return ClassificationResponse(
        classification=classification, confidence=confidence, probabilities=probabilities
    )

@app.post("/predict_batch")
def predict_batch(request: ClassificationRequestBatch, model: Model = Depends(get_model)):
    predictions_list = []
    for text in request.text_list:
        classification, confidence, probabilities = model.predict(text)

        prediction = ClassificationResponse(
            classification=classification, confidence=confidence, probabilities=probabilities
        )
        predictions_list.append({"text": text, "prediction": prediction})

    return predictions_list
