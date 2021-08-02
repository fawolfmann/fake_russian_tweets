"""Utils"""
import json
import logging
import os
from typing import Dict, Union

from bert_deploy.constants import KNOWN_CONFIGS_TYPES, KNOWN_PROBLEM_TYPES

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def read_config(config_path: Union[str, os.PathLike], mode: str) -> Dict:
    """Parse and validate configuration

    Parameters
    ----------
    config_path : Union[str, os.PathLike]
        Path for the config file.
    mode : str
        train or predict

    Returns
    -------
    Dict : dictionary with the configuration.
    """
    logger.info("Reading config: %s", config_path)
    with open(config_path, "r") as file:
        config = json.load(file)

    if mode == "train":
        _validate_config_train(config)
    elif mode == "predict":
        _validate_config_predict(config)

    return config


def _validate_config_train(config: Dict):
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


def _validate_config_predict(config: Dict):
    """Validate configs are set properly.

    Parameters
    ----------
    config : Dict
        Read configurations.

    Raises
    ------
        ValueError if any validation does not fullfil.
    """
    if not config.get("trained_model"):
        error_message = "No trained model configuration"
        logger.error(error_message)
        raise ValueError(error_message)

    if not isinstance(config["trained_model"].get("pretrained_model_name"), str):
        error_message = "pretrained_model_name configuration is not str"
        logger.error(error_message)
        raise ValueError(error_message)

    if not os.path.exists(config["trained_model"].get("pretrained_model_results")):
        error_message = "pretrained_model_results configuration does not exists"
        logger.error(error_message)
        raise ValueError(error_message)

    if not os.path.isfile(config["trained_model"].get("pretrained_model_obj_path")):
        error_message = "pretrained_model_obj_path configuration does not exists"
        logger.error(error_message)
        raise ValueError(error_message)

    if not isinstance(config["trained_model"].get("max_length"), int):
        error_message = "max_length configuration is not int"
        logger.error(error_message)
        raise ValueError(error_message)
