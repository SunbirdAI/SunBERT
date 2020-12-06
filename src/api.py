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

@app.post("/predict", response_model=Classification_Response)
def predict(request: Classification_Request, model: Model = Depends(get_model)):
    classification, confidence, probabilities = model.predict(request.text)

    return Classification_Response(
        classification = classification, confidence = confidence, probabilities=probabilities
    )