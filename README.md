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
### Data extraction and preprocessing

### Bert Tokenization

### Train 

### Store best model

## ML Pipeline - Training step
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
- CI: github workflow, nox, black, pylint.


