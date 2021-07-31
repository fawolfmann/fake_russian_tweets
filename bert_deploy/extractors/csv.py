"""CSV Data Extractor"""

import csv
import logging
import re
import string
from typing import List, Tuple, Union

import numpy as np

from bert_deploy.extractors.base import BaseBERTExtractPrepocTrain
from bert_deploy.utils import cache_extract_raw

logger = logging.getLogger(__name__)


class FakeTweetsExtractorTrain(BaseBERTExtractPrepocTrain):
    """Extractor for Amazon Reviews"""

    @cache_extract_raw()
    def extract_raw(
        self, local_path_authentic: str, local_path_fake: str
    ) -> Tuple[List, List]:
        """Read the text and labels from a csv file.

        Parameters
        ----------
        local_path_authentic : str
            local_path_authentic for .csv file to extract the authentic tweets.
        local_path_fake : str
            local_path_fake for .csv file to extract the fake tweets.

        Returns
        -------
        Tuple
            - list with the data extracted authentic tweets.
            - list with the data extracted fake tweets.
        """
        with open(local_path_authentic, mode="r", errors="ignore") as csv_file:
            loaded_dict_authentic = list(csv.DictReader(csv_file, dialect="unix"))

        with open(local_path_fake, mode="r", errors="ignore") as csv_file:
            loaded_dict_fake = list(csv.DictReader(csv_file, dialect="unix"))

        logger.info("Extraction successfull")
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
        sentences = []
        labels = []
        for raw in extracted_data[0]:
            tweet = filter_non_english_words(raw.get("text", ""))
            if tweet:
                sentences.append(tweet)
                labels.append("authentic")

        for raw in extracted_data[1]:
            tweet = filter_non_english_words(raw.get(self.sentence_col, ""))
            if tweet:
                sentences.append(tweet)
                labels.append("fake")

        logger.info("Preproccessed dataframe")

        return sentences, labels

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


def filter_non_english_words(word: str) -> Union[str, None]:
    """Remove non english words like emojis or russian letters.
    In order to use english version of bert we need it.

    Note: string.printable contains:
    0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c

    Parameters
    ----------
    word : str
        word to filter

    Returns
    -------
    [str, None]
        if word contains letter return word else None.
    """
    word = "".join(filter(lambda x: x in string.printable, word))
    if len(re.findall("\w+", word)) > 0:
        return word
    else:
        return None
