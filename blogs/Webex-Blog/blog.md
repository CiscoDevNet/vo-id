![main image](https://github.com/CiscoDevNet/vo-id/blob/master/blogs/Webex-Blog/images/main.png)
## Who said that? A quick technical intro to Speaker Diarization
Speaker Diarization answers the question: "Who spoke when?".
Currently, speakers in a meeting are identified through channel endpoints whether through PSTN or VOIP.
When speakers in the same meeting are speaking from the same room/device, they are identified as one speaker in the meeting transcript.
Because Webex Meetings recordings are provided with transcriptions, being able to answer "Who spoke when?" would allow colleagues who might have missed the meeting to quickly catch up with what was said as well as provide automatic highlights and summaries.
This is very useful, but without knowing who said what, it is more difficult for humans to skim through the content and for AI solutions to provide more accurate results.

![image of diarized meeting](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/diarized_meeting.png)
// Comment: Check with Mayada if this mock-up can be used

## Overview of our solution

1. A Fingerprint for your voice: we will discuss our approach to building the deep neural network responsible for transforming audio inputs to voice fingerprints.
2. Clustering: after transforming a sequence of audio inputs in a sequence of voice fingerprints, we'll show how we solved the problem of assigning a speaker label to each segment and group segments from the same speakers together.
3. Data pipeline: all AI models require data in order to learn the task and in this section we'll share insights on the data we have available and the strategies we adoped to label it automatically.
4. Integration with Webex: In this section we will talk about the work we've done in order to deploy the speaker diarization system to production as an additional module to our meeting transcriptions pipeline.

## Speaker Diarization in 3 steps
![Image of speaker diarization pipeline](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/speaker-diar-pipeline.png)
The process to assign speaker labels to an audio file is straightfoward and can be divided into 3 steps:
1. Split Audio: The first thing we want to do is to split the audio input into smaller audio chunks of the same length, and discard all segments that do not contain voice. We are therefore discarding silence and background noise. We use an off the shelf solution: [WebRTC Voice Activity Detector](https://github.com/wiseman/py-webrtcvad).
2. Compute Voice Fingerprints: The next step involves transforming each audio chunk to a "Voice Fingerprint". These fingerprints are 256-dimensional vectors, i.e. a list of 256 numbers. The objective is to make sure that vectors produced from different audio chunks belonging to the same speaker will be similar to each other according to some mathematical measurement. 
3. Cluster similar vectors together: The outcome of the previous step produces a list of 256-dimensional vectors for each voiced segment. The objective of this step is to group together segments that are similar to each other.

## Computing Voice Fingerprints
### The goal and the features
We do not want to restrict the quality of the diarization based on language, accent, gender or age because meetings can occur in varied settings with different microphones and background noises.
We designed the neural network responsible to compute the voice fingerprints to be robust to these factors.
This is made possible by choosing the proper neural architecture, a vast amount of training data, and data augmentation techniques.

### The architecture
The architecture of the neural network can be split into 2 parts: preprocessing and feature extraction.
The preprocessing part transforms the 1-dimensional audio input into a 2-dimensional representation.
The standard approach is to compute the [Spectrogram](https://en.wikipedia.org/wiki/Spectrogram) or the [Mel-frequency cepstral coefficients (MFCCs)](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/).
Our approach is to let the neural network learn this transformation as a sequence of 3 1-D convolutions.
The reasoning behind this choice is twofold.
First off, given enough data, our hope is that the transformation will be of higher quality for the downstream task.
The second reason is practical. We export the network to the [ONNX](https://onnx.ai/) format in order to speed up inference and as of now the operations needed to compute the MFCCs are not supported.
<br><br>
For the feature extraction we rely on a common neural network architecture commonly used for Computer Vision tasks: [ResNet18](https://arxiv.org/abs/1512.03385).
We modified the standard architecture to improve performance and to increase inference speed.

## Clustering: assigning a label to each speaker
The goal of this step is to assign a label to each audio segment so that audio segments from the same speaker gets assigned the same label.
Grouping together is done by feature similarity and is easier said than done. For example: given a pile of Lego Blocks how would we group them? It could be by color, by shape, by size, etc...
Furthermore, the objects we want to group could even have features that are not easily recognizable.
For our use case, we are trying to group together 256-dimensional vectors, in other words "objects" with 256 features, and we are counting on the neural network responsible for producing these vectors to do a good job.
At the core of each clustering algorithm there is a way to measure how similar two objects are. For our case we measure the angle between a pair of vectors: [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
The choice of this measurement is not random but is a consequence of how the neural network is trained.

Lastly, we can perform the clustering "online" or "offline".
### Online
Online clustering means that we assign a speaker label to a vector in real time as an audio chunk gets processed.
On one hand, we get a result right away and this can be useful for live captioning use cases for example.
On the other hand, we can't go back in time and correct labeling mistakes.
If the generated voice fingerprints are of high quality, the results are usually good.
We employee a straightforward greedy algorithm: as a new vector gets processed, we assign it to a new or an existing bucket (collection of vectors).
This is done by measuring how similar the new vector is to the average vector in each bucket. If it is similar enough (based on a selected similarity threshold), the vector will be added to the most similar bucket. Otherwise, it will be assigned a new bucket. <br>

![online clustering](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/online-diarize.png)

### Offline:
Offline clustering means that we assign a speaker label to each vector after we have access to the entire audio.
This allows the algorithm to go back and forth in time to find the best speaker label assignment, typically outperforming online clustering methods.
The downside is that we need to wait for the full audio recording to be available which makes this technique not suitable for real-time transcriptions.
We base our approach on [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering).
Without going into too much detail, since this is a common technique, we chose this particular method because it's robust for the particular data we have. More importantly, it is able to estimate the number of speakers automatically.
This is an important feature since we are not given the number of speaker/clusters beforehand.

## Data Pipeline

The backbone of the neural audio embedder and the clustering algorithm described above is the data used to train it and thankfully we’re in a great situation in that regard. We work closely on diarization with the Voicea team within Cisco, who are responsible for handling meeting transcription in Webex. During that process, they save audio segments from meetings that they detect speech in and make them available for download. Each saved segment is automatically labeled according to the device it comes. This allow us to use these audio segments to train our speaker embedder on a speaker recognition task. Due to the high amount of meetings that are hosted on WebEx we’re able to collect a great deal of data and the amount continues to increase over time.

One package that helps us collaborate efficiently while working with this quantity of data is DVC, short for Data Version Control. DVC is an open source tool for version control on datasets and models that are too large to track using git. When you add a file or folder to DVC, it creates a small .dvc file that tracks the original through future changes and uploads the original content to cloud storage. Changing the original file produces a new version of the .dvc file and going to older versions of the .dvc file will allow you to revert to older versions of the tracked file. We add this .dvc file to git because we can then easily switch to older versions of it and pull past versions of the tracked file from the cloud. This can be very useful in ML projects if you want to switch to older versions of datasets or identify which dataset the model was trained on,


Another benefit of DVC is functionality for sharing models and datasets. Each time one person collects data from the Voicea endpoint to add to the speaker recognition dataset, as long as that person updates the .dvc file and pushes it, the other person can seamlessly download the new data from the cloud and start training on it. The same process applies for sharing new models as well.

![image of dvc](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/dvc_version.png)


## Integration with Webex

### Meeting Recordings Page

The first WebEx integration for our diarization model was on the meeting recordings page. This page is where people can see a replay of a recorded meeting along with transcriptions provided by Voicea. Our diarization is integrated within Voicea’s meeting transcription pipeline and runs alongside it after the meeting is finished and the recording is saved. What we add to their system is a speaker label to each audio shard that Voicea identifies and segments. The goal for our system is that if we give one audio shard a label of X, the same speaker producing an audio shard later in the meeting will also receive a label of X.

There were significant efforts dedicated to improving runtime of the diarization so that it fit within an acceptable range. The biggest impact was changing the clustering to work when splitting up a meeting into smaller chunks. Because we run eigendecomposition during the spectral clustering, the runtime is O(n^3) in practice, which results in extended runtimes and memory issues for long meetings. By splitting the meeting into 20-minute chunks, running diarization on each part separately, and recombining the results, we have a slight accuracy trade off for huge reductions in runtime.


### Post Meeting Page

The other integration with WebEx is diarization for the post-meeting page. This page is shown directly after a meeting and contains transcription information as well. The main difference from the previous integration is that we have information on which device each audio segment comes from. What this means is that we can do diarization separately for each audio endpoint and avoid errors where our model predicts the same speaker for audio that came from different devices. 

This diagram shows how this works in practice. The red segments are from device 1, the blue segments are from device 2, and there are 2 speakers in each device. We first group up all the audio from each device and run the diarization separately for each group. This gives us timestamps and speaker labels within each single device grouping. We keep track of the time offset of each segment as it is grouped by device and use that to transform the speaker label times from the device grouping to what they are in the original, full meeting. 

![image of post meeting split](https://github.com/CiscoDevNet/vo-id/blob/blogs/blogs/Webex-Blog/images/post_meeting_split.png)

The post-meeting integration is also deployed within the Voicea infrastructure on Kubernetes. Our service is deployed as a Flask app within a docker image that interfaces with several other micro services. 


## Project vo-id
### We can do more:
What you can build with a good Neural Voice Embedder doesn't stop with Speaker Diarization. <br>
If you have some name-labelled audio samples from speakers that are present in the meetings you want to diarize, you could go one step further and provide the correct name for each segment.<br>
Similarly, you could build a voice authentication/verification app by comparing an audio input with a database of labelled audio segments. <br><br>
### Project vo-id
We wanted to make it easy for developers to get their hands dirty and quickly build solutions around the speaker diarization and recognition domain. Project vo-id (Voice Identification) is an open-source project structured to let developers with different expertise in AI to do so.
The [README](https://github.com/CiscoDevNet/vo-id#readme) cointains all the information needed. To give you an example, it takes only 4 lines of code to perform speaker diarization on an audio file:
```python
audio_path = "tests/audio_samples/short_podcast.wav"
from void.voicetools import ToolBox
tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
rttm = tb.diarize(audio_path)
```

### Training your own Voice Embedder
We provided a trained neural network (the vectorizer), but if you have the resources, we made it possible to update and train the neural network yourself: all the information needed is available in the [README](https://github.com/CiscoDevNet/vo-id#train-the-vectorizer)


