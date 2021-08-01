"""Utils"""
import json
import logging
import os
from typing import Dict, Union

from bert_deploy.constants import KNOWN_CONFIGS_TYPES, KNOWN_PROBLEM_TYPES

logger = logging.getLogger(__name__)


def read_config(config_path: Union[str, os.PathLike]) -> Dict:
    """Parse and validate configuration

    Parameters
    ----------
    config_path : Union[str, os.PathLike]
        Path for the config file.

    Returns
    -------
    Dict : dictionary with the configuration.
    """
    logger.info("Reading config: %s", config_path)
    with open(config_path, "r") as file:
        config = json.load(file)

    _validate_config(config)

    return config


def _validate_config(config: Dict):
    """Validate configs are set properly.

    Parameters
    ----------
    config : Dict
        Read configurations.

    Raises
    ------
        ValueError if any validation does not fullfil.
    """
    if config.get("extractor_type") not in KNOWN_CONFIGS_TYPES:
        error_message = f"Unknown extractor type, knows {KNOWN_CONFIGS_TYPES}"
        logger.error(error_message)
        raise ValueError(error_message)

    if config.get("problem_type") not in KNOWN_PROBLEM_TYPES:
        error_message = f"Unknown problem type, knows {KNOWN_PROBLEM_TYPES}"
        logger.error(error_message)
        raise ValueError(error_message)
