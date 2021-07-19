import numpy as np
from pyvad import vad

class Segmenter(object):
    def __call__(self, audio:np.array, sr:int=16000, hop_length:int=30, vad_mod:int=3) -> list:
        vact = vad(audio, sr, fs_vad=sr, hop_length=hop_length, vad_mode=vad_mod)
        segments = list()
        previous = 0
        for idx, pred in enumerate(vact):
            if previous == 0 and pred == 1:
                start = idx
                previous = 1
            if previous == 1 and pred == 0:
                end = idx
                previous = 0
                segments.append((start, end))
        if previous == 1 and pred == 1:
            end = len(vact)
            segments.append((start, end))

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