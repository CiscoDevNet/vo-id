import sys, os
import librosa
import torch
import numpy as np

from typing import Union

def vectorize(audio:Union[np.array, str], sr:int=16000, frame_stride:float=None, hop_size:float=None) -> np.array:
    """
    Parameters
    ----------
    audio : np.array or str
        1D numpy array or filepath to the audio file to vectorize.

    sr : int, optional
        Audio sample rate

    frame_stride: float, optional
        asd

    hop_size: float, optional
        asd

    Returns
    -------
    np.array
        A 2 Dimensional vector representation of the audio input.    

    """
    if isinstance(audio, str):
        if not os.path.exists(audio):
            raise FileNotFoundError(f"File not found at location: `{audio}`.")
        try:
            audio, _ = librosa.load(audio, sr=sr, mono=True)
        except Exception as e:
            raise ValueError(f"Could not read audio at location: `{audio}`.")
            
    elif not isinstance(audio, (np.ndarray, np.generic)):
        raise TypeError(f"Invalid argument type: audio should be either str or np.array.")

    audio = np.squeeze(audio)
    if not len(audio.shape) == 1:
        raise ValueError(f"Expected audio input to be 1 dimensional.")

    return audio


def diarize():
    pass


def recognize():
    pass


def verify():
    pass


if __name__ == "__main__":
    print(vectorize.__doc__)
    vectorize(123)