import numpy as np
import random
import torch
import joblib
from numpy.lib.stride_tricks import as_strided as ast

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

speaker_id_dict = joblib.load(config.get("VECTORIZER", "train_labels_dict"))
def preprocess(x):
    audios = list()
    labels = list()
    min_length = int(config.getint("AUDIO", "sr") * config.getfloat("VECTORIZER", "min_length"))
    max_length = int(config.getint("AUDIO", "sr") * config.getfloat("VECTORIZER", "max_length"))
    length = np.random.randint(min_length, max_length)

    for waveform, _, _, speaker_id, _, _ in x:
        label = speaker_id_dict.get(int(speaker_id))
        if label is None:
            raise ValueError("If you changed the trainset, you'll have to update the speaker_id map stored at `config.get('VECTORIZER', 'train_labels_dict')`")
        start = random.randint(0, max(0, waveform.shape[1] - length))
        audio = waveform[:, start:start + length]
        if audio.shape[1] < length:
            audio = torch.nn.functional.pad(audio, (0, length - audio.shape[1]), "constant", 0)
        audios.append(torch.squeeze(audio))
        labels.append(torch.tensor(label))

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

def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(data,
              shape=(num_windows,window_size*data.shape[1]),
              strides=((window_size-overlap_size)*data.shape[1]*sz,sz)
            )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows,-1,data.shape[1]))

class Full_layer(torch.nn.Module):
    '''explicitly define the fully connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = torch.nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x