"""Bert Prediction model"""
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from bert_deploy.constants import FAKE_TWEETS_ID2LABELS
from bert_deploy.prediction.base_loader import BaseModel
from bert_deploy.utils import filter_non_english_words, tokenize

logger = logging.getLogger(__name__)


class FakeTweetsModel(BaseModel):
    def __init__(self, config: Dict):
        """Initialize Bert pretained model and tokenizer.

        Parameters
        ----------
        config : Dict
            Configuration of the trained model.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])
        classifier = DistilBertForSequenceClassification.from_pretrained(
            config["pretrained_model_results"]
        )
        classifier.load_state_dict(
            torch.load(config["pretrained_model_obj_path"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

        self.max_length = config["max_length"]
        self.id2tags = FAKE_TWEETS_ID2LABELS
        logger.info("FakeTweetsModel initialized")

    def predict(self, text: str) -> Optional[Tuple]:
        """Predict the class for the given input with the initialized bert model.

        Parameters
        ----------
        text : str
            Sentences to predict.

        Returns
        -------
        Tuple
            - predicted_class: str
            - confidence of the prediction: float
            - probabilities of all the classes: List
        """
        logger.info("Input text to predict: %s", text)

        text = filter_non_english_words(text)
        if not text:
            logger.info("Text contains non english words")
            return None

        encoded_text = tokenize(self.tokenizer, text, self.max_length, "pt")

        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)
        with torch.no_grad():
            raw_output = self.classifier(input_ids, attention_mask)
            probabilities = F.softmax(raw_output.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        logger.info("Predicted output: %s", probabilities)

        return (
            self.id2tags[predicted_class],
            confidence,
            dict(zip(self.id2tags, probabilities)),
            raw_output,
        )
