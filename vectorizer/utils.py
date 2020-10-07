import numpy as np
import random
import torch

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

def preprocess(x):
    audios = list()
    labels = list()
    min_length = int(config.getint("AUDIO", "sr") * config.getfloat("VECTORIZER", "min_length"))
    max_length = int(config.getint("AUDIO", "sr") * config.getfloat("VECTORIZER", "max_length"))
    length = np.random.randint(min_length, max_length)

    for waveform, sample_rate, _, speaker_id, _, _ in x:
        start = random.randint(0, max(0, waveform.shape[1] - length))
        audio = waveform[:, start:start + length]
        if audio.shape[1] < length:
            audio = torch.nn.functional.pad(audio, (0, length - audio.shape[1]), "constant", 0)
        audios.append(torch.squeeze(audio))
        labels.append(torch.tensor(speaker_id))

    return torch.stack(audios), torch.stack(labels)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count