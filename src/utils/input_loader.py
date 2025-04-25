import json
import os
from dataclasses import dataclass
from typing import Text, Dict

import yaml

from src.utils import constants
from src.utils.custom_logger import CustomLogger


@dataclass
class InputLoader:
    def __post_init__(self):
        log_dir = os.path.join(constants.ROOT_DIR, "logs")
        self.logger = CustomLogger(
            log_dir, "logs.log", "InputLoader"
        ).logger

    def load_config(self, config_file: Text):
        _, file_extension = os.path.splitext(config_file)
        config = {}
        if "json" in file_extension.lower():
            config = self._load_json(config_file)
        if "yaml" in file_extension.lower() or "yml" in file_extension.lower():
            config = self._load_yaml(config_file)
        self.logger.debug(f"Config - {config_file} loaded successfully.")
        return config

    def load_error_messages(self, error_messages_file: Text):
        _, file_extension = os.path.splitext(error_messages_file)
        config = {}
        if "json" in file_extension.lower():
            config = self._load_json(error_messages_file)
        if "yaml" in file_extension.lower() or "yml" in file_extension.lower():
            config = self._load_yaml(error_messages_file)
        self.logger.debug(f"Error messages - {error_messages_file} loaded successfully.")
        return config


    def _load_yaml(self, yaml_file: Text):
        with open(yaml_file, "r") as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as exc:
                self.logger.error(f"Error loading YAML file: {exc}")
                return None

    def _load_json(self, json_file: Text):
        with open(json_file, "r") as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as exc:
                self.logger.error(f"Error loading JSON file: {exc}")
                return None
