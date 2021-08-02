"""Main file CLI for use this package, and usage example."""

import logging
from typing import Optional

import click

from bert_deploy.configs import read_config
from bert_deploy.prediction.tweets_loader import FakeTweetsModel

logger = logging.getLogger(__name__)


def load_model(config_path: Optional[str] = None):
    """Main function to implement Bert Prediction.

    Parameters
    ----------
    config_path : str
        path to the configuration file.
    """
    logger.info("Starting loading model")
    config_path = config_path or "./config/config_sample_fake_tweets_predict.json"
    configs = read_config(config_path, mode="predict")

    model = FakeTweetsModel(configs["trained_model"])

    return model


@click.command()
@click.option(
    "--text", type=click.STRING, help="Text to predict",
)
@click.option(
    "--config_path",
    type=click.STRING,
    help="Path to config file",
    default="./config/config_sample_fake_tweets_predict.json",
)
def predict(text: str, config_path: str = None):

    model = load_model(config_path)
    prediction = model.predict(text)

    return prediction


if __name__ == "__main__":
    predict()
