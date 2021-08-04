# Project vo-id
-----

## Project Description
This package contains the tool to build software solutions for:
* Speaker diarization
* Speaker recognition
* Speaker Verification 
* Voice Cloning
* ...

All these functionalities are possibple thanks to a neural model that converts audio into a **Voice Fingerprint** <br>
Below you can find examples on how to use the package as is. <br><br>
We are providing all the code and data for training the Neural Network, so if you have improvements to submit, fork the repo, make pull requests or open issues.

------

## Installation 

* Clone the repo:
    ```
    git clone git@github.com:CiscoDevNet/vo-id.git
    ```
* Create a Python virtual environment:
    ```
    mkdir ~/Envs/
    python3.8 -m venv ~/Envs/vo-id
    source ~/Envs/vo-id/bin/activate
    ```
* Install the package
    ```
    pip install -e .
    ```

------------

## Compute the voice fingerprint from an audio file
By default the model creates a voice fingerprint or voice vector every 100 milliseconds
```python
>>> audio_path = "tests/audio_samples/short_podcast.wav"
>>> from void.voicetools import ToolBox
>>> tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
>>> audio_vectors = tb.vectorize(audio_path)
>>> print(audio_vectors.shape)
(322, 128)
```

------

## Perform Speaker Diarization
Speaker diarization answers the question: "Who spoke when?" <br>
If you run this tool on a meeting, each spoken segment will get an anonymous speaker ID assigned. <br>
The format in use is [RTTM](https://github.com/nryant/dscore#rttm). <br>
Rich Transcription Time Marked (RTTM) files are space-delimited text files containing one turn per line, each line containing ten fields:

- ``Type``  --  segment type; should always by ``SPEAKER``
- ``File ID``  --  file name; basename of the recording minus extension (e.g.,
  ``rec1_a``)
- ``Channel ID``  --  channel (1-indexed) that turn is on; should always be
  ``1``
- ``Turn Onset``  --  onset of turn in seconds from beginning of recording
- ``Turn Duration``  -- duration of turn in seconds
- ``Orthography Field`` --  should always by ``<NA>``
- ``Speaker Type``  --  should always be ``<NA>``
- ``Speaker Name``  --  name of speaker of turn; should be unique within scope
  of each file
- ``Confidence Score``  --  system confidence (probability) that information
  is correct; should always be ``<NA>``
- ``Signal Lookahead Time``  --  should always be ``<NA>``

```python
>>> audio_path = "tests/audio_samples/short_podcast.wav"
>>> from void.voicetools import ToolBox
>>> from pprint import pprint
>>> tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
>>> diarization = tb.diarize(audio_path)
>>> pprint(diarization)
['SPEAKER filename 1 0.39 3.96 <NA> <NA> speaker0 <NA> <NA>\n',
 'SPEAKER filename 1 4.71 2.85 <NA> <NA> speaker0 <NA> <NA>\n',
 'SPEAKER filename 1 8.19 5.97 <NA> <NA> speaker0 <NA> <NA>\n',
 'SPEAKER filename 1 14.28 1.32 <NA> <NA> speaker0 <NA> <NA>\n',
 'SPEAKER filename 1 15.63 0.93 <NA> <NA> speaker0 <NA> <NA>\n',
 'SPEAKER filename 1 16.71 0.54 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 17.31 2.58 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 19.95 2.61 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 22.65 1.14 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 23.88 1.89 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 25.83 0.60 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 26.52 1.44 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 27.99 0.15 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 28.47 3.48 <NA> <NA> speaker1 <NA> <NA>\n',
 'SPEAKER filename 1 32.04 0.09 <NA> <NA> speaker1 <NA> <NA>\n']
```

## Train Vectorizer
If it's the first time you run it, this might take a while to download all the training data.<br>
Just hang on :) 
```bash
mkdir vectorizer/data
python vectorizer/train.py
```

### Notes
1. The tranining runs a classifier on the speakers, yet the speaker_ids are not consistent with how PyTorch handles the classification task:
    ```python
    max(speaker_id) > len(num_speakers) + 1
    ```
2. For this reason the file `vectorizer/speaker_ids_map.bin` stores a mapping that allows to provide labels from `0` to `num_speakers-1`
-----