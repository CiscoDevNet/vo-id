import pytest
import os
import numpy as np

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from dialogue import voicetools

toolbox = voicetools.ToolBox()
num_embeddings = config.getint("VECTORIZER", "embeddings_size")
sr = 16000

def test_check_audio_filenotfound():
    with pytest.raises(FileNotFoundError):
        toolbox._check_audio("*", sr)

def test_check_audio_valueerror():
    with pytest.raises(ValueError):
        open('tmp', 'a').close()
        toolbox._check_audio('tmp', sr)
    os.remove('tmp')

def test_check_audio_typeerror():
    with pytest.raises(TypeError):
        toolbox._check_audio(123, sr)

def test_check_audio_shapeerror():
    with pytest.raises(ValueError):
        toolbox._check_audio(np.random.randn(2,2), sr)

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
