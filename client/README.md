# Client Module
This module contains the code for face recognition, ASR (Automatic Speech Recognition), and the main behavior functions.

## Visual Module Inrtoduction
**Related File** `./Visual/detection2.py`

The vision module leverages [**InsightFace**](https://github.com/deepinsight/insightface) and [**Yolov8**](https://github.com/ultralytics/ultralytics) technologies.  InsightFace is used to extract face embeddings from user images, which are then compared with embeddings in the gallery to perform face recognition. Concurrently, Yolo tracks IDs to monitor whether individuals remain within the scene. By combining these techniques, we also achieve reliable Re-Identification (ReID).

## ASR Module 
**Related File** `./Whisper_speaker_diarization/whisper.py`

The ASR module utilizes the Whisper model for audio transcription. Initially, this repository included a diarization version intended for multiple speakers, but the current implementation is simplified for single-speaker scenariosIf your use case involves only one speaker, you might prefer using the direct [**Whisper**](https://github.com/openai/whisper) API for simplicity.

## LLM Module
**Related File** `./llmControl.py`

This script uses the OpenAI API to interact with our server managed by the LLM. It is responsible for creating and updating user profiles and processing data for returning users. After completing the main processes, it stores all profiles and related information to maintain a memory record for each individual.



## Communication Behavior
**Related File** `./client3.py`

This file includes all functions necessary for sending and receiving information via Flask to the server. It contains the main class for the **Client**, encompassing all functions. The main **ommunicate_behavior function** operates the vision module on a separate thread, allowing face recognition and tracking to continue seamlessly alongside other tasks.