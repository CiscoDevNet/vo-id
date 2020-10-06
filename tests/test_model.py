import pytest

import torch
import numpy as np
from vectorizer.model import Model, Normalize, Normalize3D

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

import sys
print(sys.path)
num_embeddings = config.getint("VECTORIZER", "embeddings_size")
batch_size = config.getint("VECTORIZER", "batch_size")
normalize3D = Normalize3D()
normalize = Normalize()

def test_model():
    model = Model()
    audio = torch.randn(batch_size, np.random.randint(16000, 32000))
    features = model(audio, train=True)
    assert features.shape == (batch_size, num_embeddings)

def test_normalize3d_valueerror():
    with pytest.raises(ValueError):
        spec = torch.randn(batch_size, np.random.randint(16000, 32000))
        normalize3D(spec)

def test_normalize3d():
    ref = torch.tensor([[[0.0000, 0.1111, 0.2222, 0.3333, 0.4444], 
                          [0.5556, 0.6667, 0.7778, 0.8889, 1.0000]]])
    X = torch.arange(10, 20).view(1, 2, 5)
    assert torch.sum(normalize3D(X)) == torch.sum(ref)

def test_noramlize_valueerror():
    with pytest.raises(ValueError):
        audio = torch.randn(16000)
        normalize(audio)

def test_normalize():
    ref = torch.tensor([[-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]])
    audio = torch.arange(-2, 3).view(1, 5)
    assert torch.allclose(normalize(audio), ref)
