import sys, os
import librosa
import torch
import numpy as np
from typing import Union, Tuple, List
from collections import defaultdict

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from vectorizer.model import Model
from vectorizer.utils import chunk_data
from void.utils import Segmenter
from spectralcluster import SpectralClusterer
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

class ToolBox(object):
    def __init__(self, use_cpu:bool=False):
        self.use_cpu = use_cpu
        self._load()

    def _load(self):
        self.model = Model()
        self.storage = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.storage = 'cpu' if self.use_cpu else self.storage
        checkpoint = torch.load(config.get('VECTORIZER', 'trained_model'), map_location=self.storage)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.storage)
        self.model.eval()
        self.segmenter = Segmenter()
        self.clusterer = SpectralClusterer(
            min_clusters=2,
            max_clusters=100,
            p_percentile=0.95,
            gaussian_blur_sigma=1.0)

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

        frame_stride = config.getfloat("AUDIO", "frame_stride") if frame_stride is None else frame_stride
        hop_size = config.getfloat("AUDIO", "hop_size") if hop_size is None else hop_size
        frame_stride = int(sr*frame_stride)
        hop_size = int(sr*hop_size)
        audio = chunk_data(audio, frame_stride, max(0, (frame_stride-hop_size)))
        audio = torch.from_numpy(np.array(audio).astype(np.float32)).to(self.storage)
        with torch.no_grad():
            features = self.model(audio)

        return features.cpu().numpy()


    def _diarize(self, audio:np.array, max_num_speakers:int) -> Tuple[List[Tuple[int, int]], np.array]:
        segments = self.segmenter(audio)
        audio_clips = [audio[s[0]:s[1]] for s in segments]
        vectors = list(map(self.vectorize, audio_clips)) 
        vectors = [item for sublist in vectors for item in sublist]
        self.clusterer.max_clusters = max_num_speakers
        labels = self.clusterer.predict(np.squeeze(np.array(vectors)))
        return segments, labels


    def diarize(self, audio:Union[np.array, str], sr:int=16000, max_num_speakers:int=30) -> List[str]:
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
        segments, labels = self._diarize(audio, max_num_speakers)
        for idx, segment in enumerate(segments):
            line = f"SPEAKER filename 1 {segment[0]/sr:.2f} {(segment[1]-segment[0])/sr:.2f} <NA> <NA> speaker{labels[idx]} <NA> <NA>\n"
            rttm.append(line)
        return rttm


    def recognize(self, audio:Union[np.array, str], enrollments:list, sr:int=16000, max_num_speakers:int=30) -> List[str]:
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
        rttm = list()

        audio = self._check_audio(audio, sr)
        enrollments = [(self._check_audio(audio, sr), label) for audio, label in enrollments]
        enrollments = [(self.vectorize(audio), label) for audio, label in enrollments]
        enrollment_vectors = list()
        for vectors, l in enrollments:
            for v in list(vectors):
                enrollment_vectors.append((v, l))

        # Compute representative vector for each label
        enrollment_dict = defaultdict(list)
        for vector, label in enrollment_vectors:
            enrollment_dict[label].append(np.squeeze(vector))
        enrollment_X, enrollment_y = zip(*[(np.mean(vectors, axis=0), label) for label, vectors in enrollment_dict.items()])

        # Run diarization
        segments, labels = self._diarize(audio, max_num_speakers)

        # Compute representative vector for each label
        segments_dict = defaultdict(list)
        for idx, vector in enumerate(vectors):
            segments_dict[labels[idx]].append(np.squeeze(vector))
        segment_X, segment_y = zip(*[(np.mean(vectors, axis=0), label) for label, vectors in segments_dict.items()])

        # Make sure we have the right shape
        enrollment_X = np.squeeze(enrollment_X)
        segment_X = np.squeeze(segment_X)
        if len(enrollment_X.shape) == 1:
            enrollment_X = enrollment_X[None, :]
        if len(segment_X.shape) == 1:
            segment_X = segment_X[None, :]

        cost = distance.cdist(np.array(enrollment_X), np.array(segment_X), metric='cosine')
        row_ind, col_ind = linear_sum_assignment(cost)
        num_solutions = len(row_ind)
        id2label = dict()
        # Map between speaker ID and provided label (if it exists)
        for sol in range(num_solutions):
            id2label[list(segment_y)[col_ind[sol]]] = list(enrollment_y)[row_ind[sol]]

        for idx, segment in enumerate(segments):
            label = id2label.get(labels[idx])
            if label is None:
                label = f"speaker{labels[idx]}"
            line = f"SPEAKER filename 1 {segment[0]/sr:.2f} {(segment[1]-segment[0])/sr:.2f} <NA> <NA> {label} <NA> <NA>\n"
            rttm.append(line)

        return rttm


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
            Similarity score --> [0, 1]

        """
        audio = self._check_audio(audio, sr)
        enrollments = [(self._check_audio(audio, sr), label) for audio, label in enrollments]
        enrollment_vector = [np.mean(self.vectorize(audio),axis=0) for audio, _ in enrollments]
        segments = self.segmenter(audio)
        audio_clips = [audio[s[0]:s[1]] for s in segments]
        vectors = list(map(self.vectorize, audio_clips)) 
        vectors = [item for sublist in vectors for item in sublist]
        audio_vector = np.mean(vectors, axis=0)
        similarity = max(0, np.mean(1-distance.cdist(audio_vector[None, :], np.array(enrollment_vector), 'cosine')))
        return similarity


if __name__ == "__main__":
    toolbox = ToolBox()
    print(toolbox.vectorize.__doc__)
    print(toolbox.diarize.__doc__)
    print(toolbox.recognize.__doc__)
    print(toolbox.verify.__doc__)