"""Main file CLI for use this package, and usage example."""

import logging

import click
import numpy as np
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from bert_deploy.configs import read_config
from bert_deploy.datasets.base import BaseBERTDataset
from bert_deploy.datasets.fake_tweets import FakeTweetsDataset
from bert_deploy.extractors.base import BaseBERTExtractPrepocTrain
from bert_deploy.extractors.csv import FakeTweetsExtractorTrain
from bert_deploy.utils import store_any

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config_path",
    type=click.STRING,
    help="Path to config file",
    default="./config/config_sample_csv.json",
)
@click.option(
    "--output_path", type=click.STRING, default="./data/", help="Path to output file"
)
def train(config_path: str, output_path: str):
    """Main function to implement Bert Training.
    from https://huggingface.co/transformers/custom_datasets.html

    Parameters
    ----------
    config_path : str
        path to the configuration file.
    output_path : str
        path to where store the output.
    """
    logger.info("Starting Train process")
    extractor: BaseBERTExtractPrepocTrain
    train_dataset: BaseBERTDataset
    val_dataset: BaseBERTDataset
    configs = read_config(config_path)
    url = ""

    if configs["extractor_type"] == "csv":
        extractor = FakeTweetsExtractorTrain(**configs["extractor_config"])
        logger.info("Created FakeTweetsExtractorTrain")
    tensor = extractor.extract_preprocess(url_or_paths=configs["url_or_paths"])
    store_name = configs["extractor_type"] + "_" + url.replace("/", "_")
    store_any(tensor, output_path, store_name)

    if configs["problem_type"] == "russian_fake_tweets":
        train_dataset = FakeTweetsDataset(tensor.train_inputs, tensor.train_labels)
        val_dataset = FakeTweetsDataset(
            tensor.validation_inputs, tensor.validation_labels
        )

    num_labels = len(np.unique(tensor.train_labels))
    model = DistilBertForSequenceClassification.from_pretrained(
        configs["extractor_config"]["pretrained_model_name_or_path"],
        num_labels=num_labels,
    )

    training_args = TrainingArguments(**configs["train_arguments"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    trainer.save_model()

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("Finished Train process")


if __name__ == "__main__":
    train()
