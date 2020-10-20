import pytest
import os
import numpy as np
import simpleder


import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from dialogue import voicetools
from dialogue.utils import rttm2simple

toolbox = voicetools.ToolBox()
num_embeddings = config.getint("VECTORIZER", "embeddings_size")
sr = 16000

audiopath = "tests/audio_samples/short_podcast.wav"
enroll_f1_path = "tests/audio_samples/enroll_fridman_1.wav"
enroll_f2_path = "tests/audio_samples/enroll_fridman_2.wav"
enroll_c1_path = "tests/audio_samples/enroll_chomsky_1.wav"
enroll_c2_path = "tests/audio_samples/enroll_chomsky_2.wav"
enroll_d1_path = "tests/audio_samples/enroll_dario_1.wav"
enroll_d2_path = "tests/audio_samples/enroll_dario_2.wav"

ground_truth = [
    ("Fridman", 0.383, 3.791),
    ("Fridman", 4.685, 7.475),
    ("Fridman", 8.693, 13.673),
    ("Chomsky", 14.213, 31.962), # 15.009
    # ("Chomsky", 15.598, 16.698),
    # ("Chomsky", 17.199, 22.562),
    # ("Chomsky", 22.984, 25.745),
    # ("Chomsky", 26.530, 27.847),
    # ("Chomsky", 28.406, 31.962),
]

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

def test_diarize():
    rttm = toolbox.diarize(audiopath, sr=sr, max_num_speakers=10)
    hyp = rttm2simple(rttm)
    der = simpleder.DER(ground_truth, hyp)
    print(f"DER: {der*100:.2f}%")
    assert der < 1.0

def test_recognize():
    rttm = toolbox.recognize(audiopath, 
                            enrollments=[
                                (enroll_c1_path, "Chomsky"), 
                                (enroll_f1_path, "Fridman"), 
                                (enroll_d1_path, "Dario"), 
                                (enroll_c2_path, "Chomsky"), 
                                (enroll_f2_path, "Fridman"), 
                                (enroll_d2_path, "Dario"),
                            ],
                            sr=sr, 
                            max_num_speakers=10
                        )
    hyp = rttm2simple(rttm)
    der = simpleder.DER(ground_truth, hyp)
    print(f"DER: {der*100:.2f}%")
    assert der < 1.0

def test_verify():
    audiopath = "tests/audio_samples/verify_fridman.wav"
    similarity = toolbox.verify(audiopath, 
                                enrollments=[
                                    (enroll_f1_path, "Fridman"),
                                    (enroll_f2_path, "Fridman"),
                                ])
    assert (similarity >= 0 and similarity <= 1)