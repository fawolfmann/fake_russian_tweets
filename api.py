"""Api entrypoint"""
from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel  # pylint-disable=no-name-in-module

from bert_deploy.predict import load_model
from bert_deploy.prediction.tweets_loader import FakeTweetsModel

app = FastAPI()


class TweetRequest(BaseModel):
    text: str


class TweetResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float


@app.post("/predict", response_model=TweetResponse)
def predict(request: TweetRequest, model: FakeTweetsModel = Depends(load_model)):
    response = model.predict(request.text)
    sentiment, confidence, probabilities = response
    return TweetResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )
