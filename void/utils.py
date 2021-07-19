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

def rttm2simple(rttm:list) -> list:
    """
    Convert rttm-like list to a simple hypothesis list compatible with simple DER library
    
    Parameters
    ----------
    rttm: list
        List of strings 
    
    Returns
    ----------
    list
        List of tuples containing unique ID, start and end time in seconds
    """
    output = list()
    for line in rttm:
        _, _, _, start, duration, _, _, label, _, _ = line.split()
        end = float(start)+float(duration)
        output.append((f"{label}", float(start), end))
    return output