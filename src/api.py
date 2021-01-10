from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()

class Classification_Request(BaseModel):
    text: str

class Classification_Response(BaseModel):
    probabilities: Dict[str, float]
    classification: str
    confidence: float

class Classification_Request_Batch(BaseModel):
    texts: List[str]

class Classification_Response_Batch(BaseModel):
    predictions: List[Classification_Response]

@app.post("/predict", response_model=Classification_Response)
def predict(request: Classification_Request, model: Model = Depends(get_model)):
    classification, confidence, probabilities = model.predict(request.text)

    return Classification_Response(
        classification = classification, confidence = confidence, probabilities=probabilities
    )

@app.post("/predict_batch", response_model=Classification_Response_Batch)
def predict_batch(request: Classification_Request_Batch, model: Model = Depends(get_model)):
    Text_predictions = request.texts
    for text in text_predictions:
        classification, confidence, probabilities = model.predict(text.text)
        text["classification"] = classification
        text["confidence"] = confidence
        text["probabilities"] = probabilities

    return Classification_Response_Batch(
        text_predictions
    )