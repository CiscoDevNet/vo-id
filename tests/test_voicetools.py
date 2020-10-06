import pytest
import os
import numpy as np

from dialogue import voicetools

def test_vectorize_filenotfound():
    with pytest.raises(FileNotFoundError):
        voicetools.vectorize("*")

def test_vectorize_valueerror():
    with pytest.raises(ValueError):
        open('tmp', 'a').close()
        voicetools.vectorize('tmp')
    os.remove('tmp')

def test_vectorize_typeerror():
    with pytest.raises(TypeError):
        voicetools.vectorize(123)

def test_vectorize_shapeerror():
    with pytest.raises(ValueError):
        voicetools.vectorize(np.random.randn(2,2))