![main image](www.www.com)
## Who said that? A quick technical intro to Speaker Diarization
Speaker Diarization answers the question: "Who spoke when?".
Currently speakers in a meeting are identified through channel endpoints whether through PSTN or VOIP.
When speakers in the same meeting are speaking from the same room/device, they are identified as one speaker in the meeting transcript.
Because Webex Meetings recordings are provided with transcriptions, being able to answer "Who spoke when?" would allow colleagues who might have missed meeting to quickly catch up with what was said, together with being able to provide automatic highlights and summaries.
This is very useful, but without knowing who said what, it is more difficult for humans to skim through the content, and for AI solutions to provide more accurate results.

![image of diarized meeting](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/diarized_meeting.png)

## Overview of our solution

1. A Fingerprint for your voice: we will discuss our approach to building the deep neural network responsible for transforming audio inputs in voice fingerprints.
2. Clustering - how to group speakers together: after transforming a sequence of audio inputs in a sequence of voice fingerprints, we'll show how we solved the problem of assigning a speaker label to each segment
3. Data pipeline: all AI models require data in order to learn the task, in this section we'll share insights on the data we have available and the strategies we adoped to label it automatically.
4. Integration with Webex: Webex meetings are transcribed using Voicea's ASR solution. In this section we will talk about the work we've done in order to deploy the Speaker Diarization to production as an extra module to the ASR pipeline

## Speaker Diarization in 3 steps
![Image of speaker diarization pipeline](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/speaker-diar-pipeline.png)
The process to assign speaker labels to an audio file is straightfoward and can be divided into 3 steps:
1. Split Audio: The first thing we want to do is to split the audio input into smaller audio chunks of the same length, and discard all those segments that do not contain voice. We are therefore discarding silence and background noise. We use an off the shelf solution: [WebRTC Voice Activity Detector](https://github.com/wiseman/py-webrtcvad)
2. Compute Voice Fingerprints: The next step involves transforming each audio chunk to "Voice Fingerprints". These fingerprints are 256 dimensional vectors each, in other words a list of 256 of numbers. The objective is to make sure that vectors produced from different audio chunks belonging to the same speaker will be similar to each other according to some mathematical measurement. 
3. Cluster similar vectors together: The outcome of the previous step produces a list of 256 dimensional vectors for each voiced segment. The objective of this step is to group together segments that are similar to each other.

## Computing Voice Fingerprints
### The goal and the features
We do not want to restrict the quality of the diarization based on language, accent, gender or age. 
Meetings could occur in different settings with different microphones and background noises.
We designed the Neural Network responbile to compute the voice fingerprints to be robust to all these difference.
This is made possible by choosing the proper neural architecture, a vast amount of training data and data augmentation techniques.

### The architecture
The architecture of the neural network can be split into 2 parts: preprocessing and features extraction.
The preprocessing part transforms the 1 dimensional audio input into a 2 dimensional representation.
The standard approach would be to compute the [Spectrogram](https://en.wikipedia.org/wiki/Spectrogram) or the [Mel-frequency cepstral coefficients (MFCCs)](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/).
Our approach is to let the neural network learn this transformation as a sequence of 3 1-D convolutions.
The reasoning behind this choice is 2-fold.
With enough data our hope is that the transformation will be of higher quality for the downstream task.
The second reason is practical: in order to speed up the inference time we export the network to the [ONNX](https://onnx.ai/) format, and as of now the operation needed to compute the MFCCs are not supported.
<br><br>
For the feature extraction we rely on a common Neural Network architecture commonly used for Computer Vision tasks: [ResNet18](https://arxiv.org/abs/1512.03385).
We modified the standard architecture to improve performance and to increase inference speed.

## Clustering: assignin a label to each speaker
The goal of this step is to assign a label to each audio segment so that the same speaker gets assigned the same one.
Grouping together is done by features similarity and it's easier said than done. For example: given a pile of Lego Blocks how would we group them? It could be by color, by shape, by size, etc...
Furthermoe, the objects we want to group could even have features that are not easily recognizable.
For our use case, we are trying to group together 256-dimensional vectors, in other words "objects" with 256 features, and we are counting on the Neural Network resposnbile to produce these vectors to be doing a good job.
At the core of each clustering algorithm there is a way to measure how similar 2 objects are. For our case we measure the angle between a pair of vectors: [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
The choice of this measurement is not random but is a consequence of how the neural network is trained.

Lastly, we can perform the clustering "online" or "offline"
### Online
Online clustering means that we assign a speaker label to a vector in real time as an audio chunk gets processed.
On one hand we get a result righ away and this can be useful for live caption use cases for example.
On the downside, we can't go back in time and correct possible mistakes.
If the generated voice fingerprints are of high quality, the results are usually good.
We adopted a straighforward greedy algorithm: as a new vector get processed, we can assign it to a new or an existing bucket (collection of vectors).
This is done by measuring how similar the new vector is to the average vector in each bucket; if similar enough (based on a particular similarity threshold), the vector will be added to the most similar bucket, otherwise it will be assigned a new one. <br>
![online clustering](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/online-diarize.png)

### Offline:
Offline clustering means that we assign a speaker label to each vector only after we have access to the entire audio.
This allows the algorithm to go back an forth in time thus finding the best speaker label assignment, and usually outperforms any online clustering methods.
The downside is that we need to wait for the all recording to be available which makes this technicque not suitable for real-time transcriptions.
We base our approach on [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering).
Without going into too much details, since this is a common techcnique, we chose this particular method because it's robust for the particular data we have, but most importantly it is able to estimate the number of speakers automatically.
This is an important feature since we are not given the number of speaker/clusters beforehand.

## Data Pipeline
blah

## Integration with Webex
blah
## Project vo-id
### We can do more:
What you can build with a good Neural Voice Embedder doesn't stop with Speaker Diarization. <br>
If you have some labelled audio samples from speakers that are present in the meetings you want to diarize, you could go one step further and provide the correct name for each segment.<br>
Similarly, you could build a voice authentication/verification app by comparing an audio input with a database of labelled audio segments. <br><br>
### Project vo-id
We wanted to make it easy for developers to get their hands dirty and quickly build solutions around the speaker diarization and recognition domain. Project vo-id (Voice Identification) is an open-source project structured to let developers with different expertise in AI to do so.
The [README](https://github.com/CiscoDevNet/vo-id#readme) cointains all the information needed, just to give you an example, it takes only 4 lines of code to perform speaker diarization on an audio file:
```python
audio_path = "tests/audio_samples/short_podcast.wav"
from void.voicetools import ToolBox
tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
audio_vectors = tb.vectorize(audio_path)
```

### Training your own Voice Embedder
We provided a trained neural network (the vectorizer), but if you have the resources, we made it possible to update and train the neural network yourself: all the information needed is available in the [README](https://github.com/CiscoDevNet/vo-id#train-the-vectorizer)


