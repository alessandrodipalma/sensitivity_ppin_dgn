import json
from pathlib import Path

def get_data_path():
    return Path(json.load(open("config.json"))["data_path"])

def get_working_path():
    return Path(json.load(open("config.json"))["working_path"])

def get_pred_data_path():
    return Path(json.load(open("config.json"))["pred_data_path"])

def get_case_study_path():
    return Path(json.load(open("config.json"))["case_study_path"])

