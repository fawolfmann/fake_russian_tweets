"""Api entrypoint"""
from typing import Dict

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def bert_explained():
    html_code = """
    <html>
        <head>
            <title>Welcome to Bert deployable</title>
        </head>
        <body>
            <h1>End point</h1>
        </body>
    </html>
    """
    return HTMLResponse(html_code)


def shape_explained(text: str):
    return text
