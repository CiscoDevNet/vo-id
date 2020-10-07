import pytest
import os
import numpy as np

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from dialogue import voicetools

toolbox = voicetools.ToolBox()
num_embeddings = config.getint("VECTORIZER", "embeddings_size")

def test_vectorize_filenotfound():
    with pytest.raises(FileNotFoundError):
        toolbox.vectorize("*")

def test_vectorize_valueerror():
    with pytest.raises(ValueError):
        open('tmp', 'a').close()
        toolbox.vectorize('tmp')
    os.remove('tmp')

def test_vectorize_typeerror():
    with pytest.raises(TypeError):
        toolbox.vectorize(123)

def test_vectorize_shapeerror():
    with pytest.raises(ValueError):
        toolbox.vectorize(np.random.randn(2,2))

def test_vectorize_noframestride():
    audio = np.random.randn(16000)
    features = toolbox.vectorize(audio)
    assert features.shape == (1, num_embeddings)

def test_vectorize_framestride():
    audio = np.random.randn(16000)
    features = toolbox.vectorize(audio, sr=16000, frame_stride=0.5)
    assert features.shape == (2, num_embeddings)

def test_vectorize_framestride():
    audio = np.random.randn(16000)
    features = toolbox.vectorize(audio, sr=16000, frame_stride=0.5, hop_size=0.25)
    assert features.shape == (3, num_embeddings)
