# Fake Russian Tweets
Fake Russian tweets analysis and prediction.

In this repo you will find an analysis and predicition model of the fake russian tweets published by [NBC](https://www.nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731)

The data is contains two parts, [tweets](http://nodeassets.nbcnews.com/russian-twitter-trolls/tweets.csv) and [users](http://nodeassets.nbcnews.com/russian-twitter-trolls/users.csv).

## Exploratory data analysis
There is a [notebook](EDA.ipynb) with all the analysis and the conclusion on how to use this data.


## Objective
The goal is create a machine learning model to predict if a tweet is fake or is not fake.
For this reason its used BERT model as a text classification problem, so it is added a not fake tweets dataset from [here](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_test.csv).

## ML Pipeline - Training step
In this step a DistilBert model is fine tuned for this problem.

### Data extraction and preprocessing
At the training step its merge the fake tweets dataset and labeled as fake and the non faked tweets and extracted and labeled as authentic.

Also non english words and emojis are removed from the dataset.

### Bert Tokenization
After the extraction and preprocessing the dataset is tokenized with DistilBert model for sequence classification problem.

### Train
For training is use PyTorch as backend of transformers and use the Trainer class.

### Store trained model
When the training is finished, is stored the model on the provided path.

## ML Pipeline - Inference step
For inference it is used the same trained model, with it configuration and available through a API build with [FastAPI](https://fastapi.tiangolo.com/).

### Data ingestion and preprocessing
The string is required to predict, it is preprocessed as the train data.

### Bert Tokenization
After preprocessing the input is tokenized with the same tokenizer who was trained the model.

### Predict
With the tokenized input the model predict the label for that tweet.

### Return the results
The predictions are return with the label and the certainty.

## Explanation
For explanation process it is use SHAP explanation package and it is create a frontend with [Voila](https://github.com/voila-dashboards/voila).

Please see this [notebook](explanation.ipynb)

## Deployment
This package is wrapped on a container and use two ports, one for the API and the other for the Voila dashboard.

For running with docker please execute:

docker-compose up --build

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