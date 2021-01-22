"""
API endpoints for text category prediction
"""

from typing import Dict, List

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()

class ClassificationRequestSingle(BaseModel):
    """
        Request object format for a single text passed in as a string
        for the '/predict' endpoint
    """
    text: str

class ClassificationRequestMultiple(BaseModel):
    """
        Request object format for a each particular text that is
        passed in part of the batch for the '/predict_batch' endpoint
    """
    id: str
    text: str

class ClassificationRequestBatch(BaseModel):
    """
        Request object format for a batch (list) of texts
        for the '/predict_batch' endpoint
    """
    text_list: List[ClassificationRequestMultiple]

class ClassificationResponse(BaseModel):
    """
        Response format for predictions
    """
    probabilities: Dict[str, float]
    classification: str
    confidence: float


@app.post("/predict", response_model=ClassificationResponse)
def predict(request: ClassificationRequestSingle, model: Model = Depends(get_model)):
    """
        Predictions for a single text passed in as a string
    """
    classification, confidence, probabilities = model.predict(request.text)

    return ClassificationResponse(
        classification=classification, confidence=confidence, probabilities=probabilities
    )

@app.post("/predict_batch")
def predict_batch(request: ClassificationRequestBatch, model: Model = Depends(get_model)):
    """
        Batch predictions for multiple texts passed in as a list of
        objects in the format: {"id": "some id", "text": "some text"}
    """
    predictions_list = []
    for item in request.text_list:
        classification, confidence, probabilities = model.predict(item.text)

        prediction = ClassificationResponse(
            classification=classification, confidence=confidence, probabilities=probabilities
        )
        predictions_list.append({"id": item.id, "text": item.text, "prediction": prediction})

    return predictions_list
