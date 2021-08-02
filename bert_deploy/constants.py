"""Constants file"""
# Fake Tweets
FAKE_TWEETS_LABLES2ID = {
    "authentic": 0,
    "fake": 1,
}

FAKE_TWEETS_ID2LABELS = {
    0: "authentic",
    1: "fake",
}
# Configs

CSV_CONFIG_TYPE = "csv"
KNOWN_CONFIGS_TYPES = [CSV_CONFIG_TYPE]

RUSSIAN_TWEETS_PROBLEM_TYPE = "russian_fake_tweets"
KNOWN_PROBLEM_TYPES = [RUSSIAN_TWEETS_PROBLEM_TYPE]
