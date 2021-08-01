"""CSV Data Extractor"""

import csv
import logging
from typing import List, Tuple

import numpy as np

from bert_deploy.constants import FAKE_TWEETS_LABLES_MAP
from bert_deploy.extractors.base import BaseBERTExtractPrepocTrain
from bert_deploy.utils import filter_non_english_words  # cache_extract_raw,

logger = logging.getLogger(__name__)


class FakeTweetsExtractorTrain(BaseBERTExtractPrepocTrain):
    """Extractor for Amazon Reviews"""

    # @cache_extract_raw()
    def extract_raw(self, url_or_paths: List[str]) -> Tuple[List, List]:
        """Read the text and labels from a csv file.

        Parameters
        ----------
        url_or_paths : List[str]
            - local_path_authentic for .csv file to extract the authentic tweets.
            - local_path_fake for .csv file to extract the fake tweets.

        Returns
        -------
        Tuple
            - list with the data extracted authentic tweets.
            - list with the data extracted fake tweets.
        """
        with open(url_or_paths[0], mode="r", errors="ignore") as csv_file:
            loaded_dict_authentic = list(csv.DictReader(csv_file, dialect="unix"))

        with open(url_or_paths[1], mode="r", errors="ignore") as csv_file:
            loaded_dict_fake = list(csv.DictReader(csv_file, dialect="unix"))

        logger.info(
            "Extraction successfull read from %s and %s",
            url_or_paths[0],
            url_or_paths[1],
        )
        return loaded_dict_authentic, loaded_dict_fake

    def preprocess(self, extracted_data: Tuple[List, List]) -> Tuple[List, List]:
        """Create two lists with the sentences and labels.
        Removing emojis and non english words.

        Parameters
        ----------
        extracted_data : Tuple[List, List]
            extracted raw data.

        Returns
        -------
        Tuple[List, List]
            - list of raw words.
            - list of raw labels.
        """
        sentences_authentic = []
        labels_authentic = []
        sentences_fake = []
        labels_fake = []
        for raw in extracted_data[0]:
            tweet = filter_non_english_words(raw.get("OriginalTweet", ""))
            if tweet:
                sentences_authentic.append(tweet)
                labels_authentic.append(FAKE_TWEETS_LABLES_MAP.get("authentic"))

        for raw in extracted_data[1]:
            tweet = filter_non_english_words(raw.get(self.sentence_col, ""))
            if tweet:
                sentences_fake.append(tweet)
                labels_fake.append(FAKE_TWEETS_LABLES_MAP.get("fake"))

        if not len(sentences_authentic) == len(labels_authentic):
            raise ValueError("Length of authentic labels and sentences mismatch")
        if not len(sentences_fake) == len(labels_fake):
            raise ValueError("Length of fake labels and sentences mismatch")
        logger.info(
            "Preproccessed data: %s fake tweets, %s authentic tweets",
            len(sentences_fake),
            len(sentences_authentic),
        )
        sentences_authentic.extend(sentences_fake)
        labels_authentic.extend(labels_fake)

        return sentences_authentic, labels_authentic

    def process_labels(self, labels: List) -> np.array:
        """Process labels as in this problem the labels are numbers from 1 to 5.
        Here just subtract 1 and ensure int type.

        Parameters
        ----------
        labels : List
            labels to process

        Returns
        -------
        np.array
            processed labels in as numpy.array.
        """
        return np.array(labels).astype(int)
