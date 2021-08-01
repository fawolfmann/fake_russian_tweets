"""Utils"""
from functools import wraps
from hashlib import sha256
import logging
from pathlib import Path
import pickle
import re
import string
from typing import Any, Union

logger = logging.getLogger(__name__)


def cache_extract_raw():
    """Cache extraction_raw results
    this wrapper hash the given name and cache in the cache_path set.
    """

    def use_cache_decorator(function):
        """Function result caching wrapper."""

        @wraps(function)
        def wrapper(*args):
            cache_path = args[0].cache_path
            cache_read = args[0].read_cache
            hashed_name = sha256((args[1]).encode()).hexdigest()
            filepath = Path(cache_path) / f"{hashed_name}.pkl"

            if cache_read and filepath.exists():
                result = from_pickle(filepath)
                logger.info("Using cached model: %s.", filepath)
            else:
                result = function(*args)
                Path.mkdir(Path(cache_path), exist_ok=True, parents=True)
                to_pickle(filepath, result)
                logger.info("Cached model to: %s.", filepath)
            return result

        return wrapper

    return use_cache_decorator


def to_pickle(filepath: Union[str, Path], obj: Any):
    """Pickle object."""
    with open(filepath, "wb") as handle:
        pickle.dump(
            obj, handle, protocol=pickle.HIGHEST_PROTOCOL,
        )


def from_pickle(filepath: Union[str, Path]):
    """Load pickled object."""
    with open(filepath, "rb") as handle:
        result = pickle.load(handle)
    return result


def store_any(obj: Any, output_path: str, name: str):
    """Store the output into a pickle object in the given path.

    Parameters
    ----------
    obj : Any
        object to store.
    output_path : str
        path to store the pickled object.
    """
    Path.mkdir(Path(output_path), exist_ok=True, parents=True)

    output_filepath = Path(output_path) / f"{name}_bert_extraction_tensor.pkl"
    to_pickle(output_filepath, obj)


def filter_non_english_words(word_input: str) -> Union[str, None]:
    """Remove non english words like emojis or russian letters.
    In order to use english version of bert we need it.

    Note: string.printable contains:
    0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c

    Parameters
    ----------
    word_input : str
        word to filter

    Returns
    -------
    [str, None]
        if word contains letter return word else None.
    """

    word = "".join(filter(lambda x: x in string.printable, word_input))
    if len(re.findall("\w+", word)) > 0:
        return word
    else:
        logger.warning("Removed word %s", word_input)
        return None
