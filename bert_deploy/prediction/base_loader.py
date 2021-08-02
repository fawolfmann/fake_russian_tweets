"""Bert Prediction model"""
from abc import ABC
from typing import Dict, Optional, Tuple


class BaseModel(ABC):
    def __init__(self, config: Dict):
        """Initialize model and attributes.

        Parameters
        ----------
        config : Dict
            Configuration of the trained model.
        """

    def predict(self, text: str) -> Optional[Tuple]:
        """Predict the class for the given input with the initialized model.

        Parameters
        ----------
        text : str
            Sentences to predict.

        Returns
        -------
        Optional[Tuple]
            - predicted_class: str
            - confidence of the prediction: float
            - probabilities of all the classes: List
        """
        return None
