import sys, os
import librosa
import torch
import numpy as np
from typing import Union

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from vectorizer.model import Model
from vectorizer.utils import chunk_data
from dialogue.utils import Segmenter
from spectralcluster import SpectralClusterer

class ToolBox(object):
    def __init__(self):
        self._load()

    def _load(self):
        self.model = Model()
        self.storage = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(config.get('VECTORIZER', 'trained_model'), map_location=self.storage)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.storage)
        self.model.eval()
        self.segmenter = Segmenter()
        self.clusterer = SpectralClusterer(
            min_clusters=2,
            max_clusters=100,
            p_percentile=0.95,
            gaussian_blur_sigma=1)

    def _check_audio(self, audio:Union[np.array, str], sr:int) -> Union[np.array, str]:
        if isinstance(audio, str):
            if not os.path.exists(audio):
                raise FileNotFoundError(f"File not found at location: `{audio}`.")
            try:
                audio, _ = librosa.load(audio, sr=sr, mono=True)
            except Exception as e:
                raise ValueError(f"Exception: {e}\nCould not read audio at location: `{audio}`.")
                
        elif not isinstance(audio, (np.ndarray, np.generic)):
            raise TypeError(f"Invalid argument type: audio should be either str or np.array.")

        audio = np.squeeze(audio)
        if not len(audio.shape) == 1:
            raise ValueError(f"Expected audio input to be 1 dimensional.")
        return audio


    def vectorize(self, audio:Union[np.array, str], sr:int=16000, frame_stride:float=None, hop_size:float=None) -> np.array:
        """
        Parameters
        ----------
        audio : np.array or str
            1D numpy array or filepath to the audio file to vectorize.

        sr : int, optional
            Audio sample rate

        frame_stride: float, optional
            Chunk audio in frames of length frame_stride seconds

        hop_size: float, optional
            Chunk audio in frames of length frame_stride seconds with hop_size seconds


        Returns
        -------
        np.array
            A 2 Dimensional vector representation of the audio input.    

        """
        audio = self._check_audio(audio, sr)

        if frame_stride is not None:
            frame_stride = int(sr*frame_stride)
            if hop_size is not None:
                hop_size = int(sr*hop_size)
            else:
                hop_size = frame_stride
            audio = chunk_data(audio, frame_stride, max(0, (frame_stride-hop_size)))
            print(audio.shape)

        else:
            audio = audio[None, :]
        
        audio = torch.from_numpy(np.array(audio).astype(np.float32)).to(self.storage)
        with torch.no_grad():
            features = self.model(audio)

        return features.cpu().numpy()


    def diarize(self, audio:Union[np.array, str], sr:int=16000, max_num_speakers:int=30) -> list:
        """
        Parameters
        ----------
        audio : np.array or str
            1D numpy array or filepath to the audio file to vectorize.

        sr : int, optional
            Audio sample rate

        max_num_speakers: int, optional
            Maximum amount of expected speakers in the audio

        Returns
        -------
        list
            A list of strings. Each line is compatible with the RTTM format

        """
        rttm = list()

        audio = self._check_audio(audio, sr)
        segments = self.segmenter(audio)
        audio_clips = [audio[s[0]:s[1]] for s in segments]
        vectors = list(map(self.vectorize, audio_clips)) 
        self.clusterer.max_clusters = max_num_speakers
        labels = self.clusterer.predict(np.squeeze(np.array(vectors)))
        for idx, segment in enumerate(segments):
            line = f"SPEAKER filename 1 {segment[0]/sr:.2f} {(segment[1]-segment[0])/sr:.2f} <NA> <NA> speaker{labels[idx]} <NA> <NA>\n"
            rttm.append(line)
        return rttm


    def recognize(self, audio:Union[np.array, str], enrollments:list, sr:int=16000, max_num_speakers:int=30) -> list:
        """
        Parameters
        ----------
        audio : np.array or str
            1D numpy array or filepath to the audio file to vectorize.

        enrollments: list
            list of tuples: (audio:Union[np.array, str], label:str)

        sr : int, optional
            Audio sample rate

        max_num_speakers: int, optional
            Maximum amount of expected speakers in the audio

        Returns
        -------
        list
            A list of strings. Each line is compatible with the RTTM format

        """
        audio = self._check_audio
        enrollments = [(self._check_audio(audio, sr), label) for audio, label in enrollments]
        return list()


    def verify(self, audio:Union[np.array, str], enrollments:list, sr:int=16000 ) -> float:
        """
        Parameters
        ----------
        audio : np.array or str
            1D numpy array or filepath to the audio file to vectorize.

        enrollments: list
            list of tuples: (audio:Union[np.array, str], label:str)

        sr : int, optional
            Audio sample rate

        Returns
        -------
        float
            Confidence score

        """
        audio = self._check_audio
        enrollments = [(self._check_audio(audio, sr), label) for audio, label in enrollments]
        return 0.5


if __name__ == "__main__":
    toolbox = ToolBox()
    print(toolbox.vectorize.__doc__)