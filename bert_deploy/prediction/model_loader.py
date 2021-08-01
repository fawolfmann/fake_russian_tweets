"""Bert Prediction model"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from bert_deploy.constants import FAKE_TWEETS_LABLES_MAP
from bert_deploy.utils import filter_non_english_words


class FakeTweetsModel:
    def __init__(self, config: Dict):
        """Initialize Bert pretained model and tokenizer.

        Parameters
        ----------
        config : Dict
            Configuration of the trained model.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config["BERT_MODEL"])
        classifier = DistilBertForSequenceClassification.from_pretrained(
            config["BERT_MODEL"]
        )
        classifier.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

        self.max_length = config["max_length"]
        self.id2tags = FAKE_TWEETS_LABLES_MAP

    def predict(self, text: str) -> Optional[Dict]:
        """Predict the class for the given input with the initialized bert model.

        Parameters
        ----------
        text : str
            Sentences to predict.

        Returns
        -------
        Dict
            - predicted_class: str
            - confidence of the prediction: float
            - probabilities of all the classes: List
        """
        text = filter_non_english_words(text)

        if not text:
            return None

        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)
        with torch.no_grad():
            probabilities = F.softmax(
                self.classifier(input_ids, attention_mask).logits, dim=1
            )
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        return (
            self.id2tags[predicted_class],
            confidence,
            dict(zip(self.id2tags, probabilities)),
        )
