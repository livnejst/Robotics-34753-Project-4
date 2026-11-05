import configparser
import os
import sys

DEFAULT_CONFIG_FILE = "config.ini"


class Config:
    def __init__(self, config_file=None):
        if config_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, DEFAULT_CONFIG_FILE)

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.index = int(self.config.get("camera", "index"))

        self.is_win_os = sys.platform.startswith("win")

        self.device_name = self.config.get("robot", "device_name")


if __name__ == "__main__":
    config = Config()
    print(f"Camera index: {config.index}")
    print(f"Robot device name: {config.device_name}")
