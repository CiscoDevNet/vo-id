import torch
import torchaudio

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("../config.ini")

from vectorizer.model import Model
def main():
    model = Model()