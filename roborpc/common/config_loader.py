# import argparse
import argparse
import os
# from typing import Any

import tomli as tomllib
from pathlib import Path


class _ConfigLoader:

    def __init__(self) -> None:
        self.common_directory = Path(__file__).parent
        config_path_name = os.path.join(self.common_directory, 'config', 'configuration.toml')
        with open(config_path_name, 'rb') as f:
            self.config = tomllib.load(f)


config = _ConfigLoader().config
