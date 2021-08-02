"""Explainability Bert Module"""

import numpy as np
import scipy as sp
import shap

from bert_deploy.predict import load_model

model = load_model()


def predict_helper(text: str):
    prediction = model.predict(text)

    scores = (np.exp(prediction[3]).T / np.exp(prediction[3]).sum(-1)).T
    val = sp.special.logit(scores[:, 1])  # pylint: disable=no-member

    return val


def explainable_input(input_text: str):
    explainer = shap.Explainer(predict_helper, model.tokenizer)
    shap_values = explainer(input_text, fixed_context=1)
    return shap.plots.text(shap_values)
