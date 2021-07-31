"""Main file CLI for use this package, and usage example."""

import click
import numpy as np
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from bert_deploy.configs import read_config
from bert_deploy.datasets.base import BaseBERTDataset
from bert_deploy.datasets.covid_tweets import CovidDataset
from bert_deploy.extractors.base import BaseBERTExtractPrepoc
from bert_deploy.extractors.csv import CSVExtractor
from bert_deploy.utils import store_any


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
    """Main function to implement Bert Deploy.
    from https://huggingface.co/transformers/custom_datasets.html

    Parameters
    ----------
    config_path : str
        path to the configuration file.
    output_path : str
        path to where store the output.
    """
    extractor: BaseBERTExtractPrepoc
    train_dataset: BaseBERTDataset
    val_dataset: BaseBERTDataset
    configs = read_config(config_path)
    url = ""

    if configs["extractor_type"] == "csv":
        extractor = CSVExtractor(**configs["extractor_config"])
        url = configs.get("extractor_url", "")

    tensor = extractor.extract_preprocess(url)
    store_name = configs["extractor_type"] + "_" + url.replace("/", "_")
    store_any(tensor, output_path, store_name)

    if configs["problem_type"] == "covid_tweets":
        train_dataset = CovidDataset(tensor.train_inputs, tensor.train_labels)
        val_dataset = CovidDataset(tensor.validation_inputs, tensor.validation_labels)

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

    trainer.train()
    model_name = "trained_distil_bert_covid_tweets"
    store_any(model, output_path, model_name)


if __name__ == "__main__":
    train()
