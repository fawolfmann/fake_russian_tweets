# Fake Russian Tweets
Fake Russian tweets analysis and prediction.

In this repo you will find an analysis and predicition model of the fake russian tweets published by [NBC](https://www.nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731)

The data is contains two parts, [tweets](http://nodeassets.nbcnews.com/russian-twitter-trolls/tweets.csv) and [users](http://nodeassets.nbcnews.com/russian-twitter-trolls/users.csv).

## Exploratory data analysis
There is a [notebook] with all the analysis. and the conclusion on how to create a model.


## Objective
The goal is create a machine learning model to predict if a tweet is fake or is not fake.
For this reason its used BERT model as a text classification problem, so it is added a not fake tweets dataset from [here]().

## ML Pipeline - Training step
In this step a DistilBert model is fine tuned for this problem.

### Data extraction and preprocessing
At the training step its merge the fake tweets dataset and labeled as fake and the non faked tweets and extracted and labeled as authentic.

Also non english words and emojis are removed from the dataset.

### Bert Tokenization
After the extraction and preprocessing the dataset is tokenized with DistilBert model for sequence classification problem.

### Train
For training is use PyTorch as backend of transformers and use the Trainer class.

### Store best model
When the training is finished, is stored the best fitted model.

## ML Pipeline - Inference step
### Data ingestion and preprocessing

### Bert Tokenization

### Predict

### Show the results and plot

## Tools

### Package tools
- Transformes
- PyTorch
- FastAPI
- SHAP for BERT
- Docker & docker-compose

### Developing tools
- Poetry (dependencies and environment management)
- CI: github workflow, nox, black, pylint, isort, pre-commit.
- Gitflow.
- Pytests for testing

## Installation

## How to use it