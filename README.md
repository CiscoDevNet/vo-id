# Project vo-id
-----
## Train Vectorizer
If it's the first time you run it, this might take a while to download all the training data.<br>
Just hang on :) 
```bash
mkdir vectorizer/data
python vectorizer.train.py
```

### Notes
1. The tranining runs a classifier on the speakers, yet the speaker_ids are not consistent with how PyTorch handles the classification task:
    ```python
    max(speaker_id) > len(num_speakers) + 1
    ```
2. For this reason the file `vectorizer/speaker_ids_map.bin` stores a mapping that allows to provide labels from `0` to `num_speakers-1`
-----