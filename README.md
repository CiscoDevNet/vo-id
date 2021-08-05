# Project vo-id
-----

## Project Description
* [Installation](#installation)
* [Voice Fingerprinting](#compute-the-voice-fingerprint-from-an-audio-file)
* [Speaker diarization](#perform-speaker-diarization)
* [Speaker recognition](#perform-speaker-recognition)
* [Speaker Verification](#perform-speaker-verification) 
* [Voice Cloning](#voice-cloning)

All these functionalities are possibple thanks to a neural model that converts audio into a **Voice Fingerprint** <br>
Below you can find examples on how to use the package as is. <br><br>
We are providing all the code and data for training the Neural Network, so if you have improvements to submit, please fork the repo, make pull requests or open issues. :handshake:

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
By default the model creates a voice fingerprint or voice vector every 100 milliseconds.
```python
audio_path = "tests/audio_samples/short_podcast.wav"
from void.voicetools import ToolBox
tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
audio_vectors = tb.vectorize(audio_path)
print(audio_vectors.shape)
# (322, 128)
```

------

## Perform Speaker Diarization
Speaker diarization answers the question: ``"Who spoke when?"``. <br>
If you run this tool on a meeting recording for example, each spoken segment will get an anonymous speaker ID assigned. <br>
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
audio_path = "tests/audio_samples/short_podcast.wav"
from void.voicetools import ToolBox
from pprint import pprint
tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
rttm = tb.diarize(audio_path)
pprint(rttm)
""" 
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
"""
```

-----

## Perform Speaker Recognition
Speaker Recognition works very similarly to Speaker Diarization, with the difference that each voice segment gets assigned the name of the person the system thinks it's speaking. <br>
In order to do so we need to provide ``Enrollment file``, meaning audio files with examples of the voice of the speakers present in the audio we are diarizing.

```python
audio_path = "tests/audio_samples/short_podcast.wav"
# Provide enrollment samples
enroll_f1_path = "tests/audio_samples/enroll_fridman_1.wav"
enroll_f2_path = "tests/audio_samples/enroll_fridman_2.wav"
enroll_c1_path = "tests/audio_samples/enroll_chomsky_1.wav"
enroll_c2_path = "tests/audio_samples/enroll_chomsky_2.wav"
enroll_d1_path = "tests/audio_samples/enroll_dario_1.wav"
enroll_d2_path = "tests/audio_samples/enroll_dario_2.wav"
from void.voicetools import ToolBox
from pprint import pprint
tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
rttm = tb.recognize(audio_path, 
                            enrollments=[
                                (enroll_c1_path, "Chomsky"), 
                                (enroll_f1_path, "Fridman"), 
                                (enroll_d1_path, "Dario"), 
                                (enroll_c2_path, "Chomsky"), 
                                (enroll_f2_path, "Fridman"), 
                                (enroll_d2_path, "Dario"),
                            ],
                            max_num_speakers=10
                        )
pprint(rttm)
"""
['SPEAKER filename 1 0.39 3.96 <NA> <NA> Chomsky <NA> <NA>\n',
 'SPEAKER filename 1 4.71 2.85 <NA> <NA> Chomsky <NA> <NA>\n',
 'SPEAKER filename 1 8.19 5.97 <NA> <NA> Chomsky <NA> <NA>\n',
 'SPEAKER filename 1 14.28 1.32 <NA> <NA> Chomsky <NA> <NA>\n',
 'SPEAKER filename 1 15.63 0.93 <NA> <NA> Chomsky <NA> <NA>\n',
 'SPEAKER filename 1 16.71 0.54 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 17.31 2.58 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 19.95 2.61 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 22.65 1.14 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 23.88 1.89 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 25.83 0.60 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 26.52 1.44 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 27.99 0.15 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 28.47 3.48 <NA> <NA> Fridman <NA> <NA>\n',
 'SPEAKER filename 1 32.04 0.09 <NA> <NA> Fridman <NA> <NA>\n']
"""
```

#### NB: Even though we provided `3 enrollment speakers`, becacuse the meeting only had 2, the system correctly outputs only `2` in total.
---

## Perform Speaker Verification
We can use our voice similarly to how we use our fingerprints or faces on modern smartphones: to let only the right users have access to a system. <br>
By providing voice examples of someone's voice, we can then compare new audio samples with the ones we have previously stored.

```python
enroll_f1_path = "tests/audio_samples/enroll_fridman_1.wav"
enroll_f2_path = "tests/audio_samples/enroll_fridman_2.wav"
new_audio = "tests/audio_samples/verify_fridman.wav"
from void.voicetools import ToolBox
tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
similarity = tb.verify(new_audio, 
                                enrollments=[
                                    (enroll_f1_path, "Fridman"),
                                    (enroll_f2_path, "Fridman"),
                                ])
print(f"Same person probability: {similarity*100:.2f}%")
#Same person probability: 82.24%
```

--------

## Voice Cloning
Work in Progress...

--------

## Train the Vectorizer
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
