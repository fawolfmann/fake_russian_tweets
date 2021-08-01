"""Constants file"""
# COVID sentiment analysis
FAKE_TWEETS_LABLES_MAP = {
    "authentic": 0,
    "fake": 1,
}

# Configs

CSV_CONFIG_TYPE = "csv"
KNOWN_CONFIGS_TYPES = [CSV_CONFIG_TYPE]

RUSSIAN_TWEETS_PROBLEM_TYPE = "russian_fake_tweets"
KNOWN_PROBLEM_TYPES = [RUSSIAN_TWEETS_PROBLEM_TYPE]
