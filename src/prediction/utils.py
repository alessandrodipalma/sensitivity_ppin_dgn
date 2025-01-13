import json
from pathlib import Path

def get_data_path():
    return Path(json.load(open("config.json"))["data_path"])

def get_working_path():
    return Path(json.load(open("config.json"))["working_path"])
