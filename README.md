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