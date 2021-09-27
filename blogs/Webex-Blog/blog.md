![main image](www.www.com)
## Who said that? A quick technical intro to Speaker Diarization
Speaker Diarization answers the question: "Who spoke when?".
Currently speakers in a meeting are identified through channel endpoints whether through PSTN or VOIP.
When speakers in the same meeting are speaking from the same room/device, they are identified as one speaker in the meeting transcript.
Because Webex Meetings recordings are provided with transcriptions, being able to answer "Who spoke when?" would allow colleagues who might have missed meeting to quickly catch up with what was said, together with being able to provide automatic highlights and summaries.
This is very useful, but without knowing who said what, it is more difficult for humans to skim through the content, and for AI solutions to provide more accurate results.

![image of diarized meeting](lksj)

## Overview of our solution

1. A Fingerprint for your voice: we will discuss our approach to building the deep neural network responsible for transforming audio inputs in voice fingerprints.
2. Clustering - how to group speakers together: after transforming a sequence of audio inputs in a sequence of voice fingerprints, we'll show how we solved the problem of assigning a speaker label to each segment
3. Data pipeline: all AI models require data in order to learn the task, in this section we'll share insights on the data we have available and the strategies we adoped to label it automatically.
4. Integration with Webex: Webex meetings are transcribed using Voicea's ASR solution. In this section we will talk about the work we've done in order to deploy the Speaker Diarization to production as an extra module to the ASR pipeline

## Extracting Voice Fingerprints
* The goal and the features: blah
* The architecture: blah
* The triplet loss: blah
## Online and Offline methods
* Online:
* Offline:

## Data Pipeline
blah

## Integration with Webex
blah
## Project vo-id
* We can do more:
What you can build with a good Neural Voice Embedder doesn't stop with Speaker Diarization. <br>
If you have some labelled audio samples from speakers that are present in the meetings you want to diarize, you could go one step further and provide the correct name for each segment.<br>
Similary you could build a voice authentication/verification app by comparing an audio input with a database of labelled audio segments. <br><br>
* Project vo-id: we wanted to make it easy for developers to get their hands dirty and quickly build solutions around the speaker diarization and recognition domain. Project vo-id (Voice Identification) is structured to let developers with different expertise in AI to do so.
<br>
The [README](https://github.com/CiscoDevNet/vo-id#readme) cointains all the information needed, just to give you an example, it takes only 4 lines of code to perform speaker diarization on an audio file:
    ```python
    audio_path = "tests/audio_samples/short_podcast.wav"
    from void.voicetools import ToolBox
    tb = ToolBox(use_cpu=True) # Leave `use_cpu` blank to let the machine use the GPU if available  
    audio_vectors = tb.vectorize(audio_path)
    ```

* Training your own Voice Embedder: we provided a trained neural network (the vectorizer), but if you have the resources, we made it possible to update and train the neural network yourself: all the information needed is available in the [README](https://github.com/CiscoDevNet/vo-id#train-the-vectorizer)


