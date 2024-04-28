# import whisper
from faster_whisper import WhisperModel
import datetime
import pandas as pd
import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import torch

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from gpuinfo import GPUInfo

import wave
import contextlib
from transformers import pipeline
import psutil

# import Recroding video & audio
import os


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


class Whisper:
    def __init__(self, MODEL_NAME="vumichien/whisper-medium-jp", whisper_model="medium", gpu_id=0):
        # whisper_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
        # MODEL_NAME = "vumichien/whisper-medium-jp"

        #self.model = WhisperModel(whisper_model, compute_type="int8")
        self.model = WhisperModel(whisper_model, device="cuda", compute_type="int8_float16")
        self.MODEL_NAME = MODEL_NAME
        lang = "en"
        device = gpu_id if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_NAME,
            chunk_length_s=30,
            device=device,
        )
        os.makedirs('output', exist_ok=True)
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language=lang,
                                                                                               task="transcribe")

        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def transcribe(self, microphone, file_upload):
        warn_output = ""
        if (microphone is not None) and (file_upload is not None):
            warn_output = (
                "WARNING: You've uploaded an audio file and used the microphone. "
                "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
            )

        elif (microphone is None) and (file_upload is None):
            return "ERROR: You have to either use the microphone or upload an audio file"

        file = microphone if microphone is not None else file_upload

        text = self.pipe(file)["text"]

        return warn_output + text

    def speech_to_text(self, audio_file, save_path,selected_source_lang, joined_person):
        """
        1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
        2. Generating speaker embeddings for each segments.
        3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.

        Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
        Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
        """

        # model = whisper.load_model(whisper_model)
        # model = WhisperModel(whisper_model, device="cuda", compute_type="int8_float16")

        # model = WhisperModel(whisper_model, compute_type="int8")
        # update number of speakers
        num_speakers = len(joined_person) if joined_person is not None else 0

        print("Whisper Model is set up")
        time_start = time.time()
        if (audio_file == None):
            raise ValueError("Error no audio input")
        print(f'Loading the audio file: ',audio_file)

        try:
            # Get duration
            with contextlib.closing(wave.open(audio_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            print(f"conversion to wav ready, duration of audio file: {duration}")

            # Transcribe audio
            options = dict(language=selected_source_lang, beam_size=5, best_of=5)
            transcribe_options = dict(task="transcribe", **options)
            segments_raw, info = self.model.transcribe(audio_file, **transcribe_options)
            # Convert back to original openai format
            segments = []
            i = 0
            for segment_chunk in segments_raw:
                chunk = {}
                chunk["start"] = segment_chunk.start
                chunk["end"] = segment_chunk.end
                chunk["text"] = segment_chunk.text
                segments.append(chunk)
                i += 1
            #print("transcribe audio done with fast whisper")
            if len(segments) == 0:
                print("Nothing containing in this audio")
                return
        except Exception as e:
            raise RuntimeError("Error converting video to audio")

        try:
            # Create embedding
            def segment_embedding(segment):
                audio = Audio()
                start = segment["start"]
                # Whisper overshoots the end timestamp in the last segment
                end = min(duration, segment["end"])
                clip = Segment(start, end)
                waveform, sample_rate = audio.crop(audio_file, clip)
                return self.embedding_model(waveform[None])

            embeddings = np.zeros(shape=(len(segments), 192))
            for i, segment in enumerate(segments):
                embeddings[i] = segment_embedding(segment)
            embeddings = np.nan_to_num(embeddings)
            #print(f'Embedding shape: {embeddings.shape}')

            # if there is no target person who has be detected in this conversation, let the algorithm evaluate by
            # itself to find the #spekears
            if num_speakers == 0:  # no target joined
                # Find the best number of speakers
                try:
                    score_num_speakers = {}

                    for num_speakers in range(2, 10 + 1):  # simple change the range will not Work, we need to use
                        # try. and whenever the issue popped up, set it as one speaker.
                        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                        score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                        score_num_speakers[num_speakers] = score
                    best_num_speaker = max(score_num_speakers, key=lambda x: score_num_speakers[x])
                    print(
                        f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
                except:
                    best_num_speaker = 1
            else:
                best_num_speaker = num_speakers

            if best_num_speaker == 1:
                if num_speakers == 0:
                    # if there is no speaker has been detected in this conversation
                    for i in range(len(segments)):
                        segments[i]["speaker"] = 'SPEAKER ' + str(i + 1)
                else:
                    # num_speakers == 1
                    for i in range(len(segments)):
                        segments[i]["speaker"] = 'SPEAKER ' + joined_person[0]
            else:
                try:  # if the best_number_speaker is greater than 1, but have only 1 person talk for this part of audio
                    # Assign speaker label
                    clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
                    labels = clustering.labels_
                    print("Is over the clustering")
                    for i in range(len(segments)):
                        segments[i]["speaker"] = 'SPEAKER ' + joined_person[int(labels[i] + 1)]
                except:
                    print("label wrong -_-")
                    for i in range(len(segments)):
                        # default use the first person at this stage
                        segments[i]["speaker"] = 'SPEAKER ' + joined_person[0]
                        # segments[i]["speaker"] = 'SPEAKER ' + str(i + 1)

            txtFilename = "./output/exchange_information/py_to_cpp.txt"
            # Make output
            objects = {
                'Start': [],
                'End': [],
                'Speaker': [],
                'Text': []
            }
            text = ''
            # print(segments)
            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    objects['Start'].append(str(convert_time(segment["start"])))
                    objects['Speaker'].append(segment["speaker"])
                    if i != 0:
                        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                        objects['Text'].append(text)
                        text = ''
                text = text + segment["speaker"] + ' say: ' + '\'' + segment["text"] + '\'' + '     '
                # with open(txtFilename, 'a') as file:
                #     file.write(text)

            tempValue = 0 if i == 0 else i-1
            #print(f"i is. {i}")
            #print(f"tempValue is , {tempValue}")
            #print(f"segments is , {segments}")
            #print(f"objects is , {objects}")
            objects['End'].append(str(convert_time(segments[tempValue]["end"])))
            objects['Text'].append(text)

            time_end = time.time()
            time_diff = time_end - time_start
            memory = psutil.virtual_memory()
            gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
            gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
            gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
            system_info = f"""
            *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
            *Processing time: {time_diff:.5} seconds.*
            *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
            """
            # save_path = "./../database/transcripts/transcript_result.csv"
            df_results = pd.DataFrame(objects)
            df_results.to_csv(save_path, mode='a')
            print(f"Result has been saved to [Save path]: {os.path.abspath(save_path)}")
            print(df_results)
            # print(system_info)
            return text

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)


