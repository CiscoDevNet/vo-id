from inaSpeechSegmenter import Segmenter as Seg
import soundfile as sf
import numpy as np
import os

class Segmenter(object):
    def __init__(self):
        self.seg = Seg()
        
    def __call__(self, audio:np.array, sr:int=16000) -> list:
        temp = "/tmp/asd.wav"
        sf.write(temp, audio[:, None], sr)
        segments = self.seg(temp)
        os.remove(temp)
        segments = [(int(segment[1]*sr), int(segment[2]*sr)) for segment in segments if segment[0] in ["male", "female"]]
        return segments