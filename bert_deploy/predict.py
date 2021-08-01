"""Main file CLI for use this package, and usage example."""

import logging

import click

from bert_deploy.configs import read_config
from bert_deploy.prediction.model_loader import FakeTweetsModel

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config_path",
    type=click.STRING,
    help="Path to config file",
    default="./config/config_sample_fake_tweets_predict.json",
)
def load_model(config_path: str):
    """Main function to implement Bert Prediction.

    Parameters
    ----------
    config_path : str
        path to the configuration file.
    """
    logger.info("Starting loading model")
    configs = read_config(config_path)

    model = FakeTweetsModel(configs["trained_model"])

    return model


def predict(text: str, config_path: str = None):
    config_path = config_path or "./config/config_sample_fake_tweets_predict.json"
    model = load_model(config_path)
    prediction = model.predict(text)

    return prediction


if __name__ == "__main__":
    predict()
