# Client Module
This module consist the code for the function for face recognition, ASR and the main behavior function. 

## Visual Module Inrtoduction
**Relate File** `./Visual/detection2.py`

The vision module is powered by [**InsightFace**](https://github.com/deepinsight/insightface) and [**Yolov8**](https://github.com/ultralytics/ultralytics). We use inisghtFace to extract the face embedding from user face on the image, and do the compraision with the face_embedding in the gallary to complete the face recognition. Meanwhile, we use Yolo with the ID to represent that the person still in the scene and have not leave. By conbining these techniques. We complete the ReID also. 

## ASR Module 
**Relate File** `./Whisper_speaker_diarization/whisper.py`

We use Whisper Module to do the audio transcription. But this module in this reposuitory have some redundent part, because I tried to use the Diarization version at first, but the current work only do for one person. You can directly and simply use the [**Whisper**](https://github.com/openai/whisper) API if you only have one person speaking. 

## Communication Behavior
**Related File** `./client3.py`

This file contain the all function that receive and send information by Flask to server. Meanwhile, it coantin the main class for **Client** with all functions. In the main communicate_behavior function, we call the vision and otehr function seperatelym the vision module is running on a single thread, which made the recognition and tracking can keep running when we running the other sections. 