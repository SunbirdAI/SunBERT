from typing import Dict, List

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
    text_list: List[str]

class Classification_Response_Batch(BaseModel):
    predictions: List

@app.post("/predict", response_model=Classification_Response)
def predict(request: Classification_Request, model: Model = Depends(get_model)):
    classification, confidence, probabilities = model.predict(request.text)

    return Classification_Response(
        classification = classification, confidence = confidence, probabilities=probabilities
    )

@app.post("/predict_batch")
def predict_batch(request: Classification_Request_Batch, model: Model = Depends(get_model)):
    predictions = []
    for text in request.text_list:
        classification, confidence, probabilities = model.predict(text)
    
        predicted = Classification_Response(
            classification = classification, confidence = confidence, probabilities=probabilities
        )
        predictions.append(predicted)

    return predictions
