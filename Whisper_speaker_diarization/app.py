# import whisper
from faster_whisper import WhisperModel
import datetime
import subprocess
import gradio as gr
from pathlib import Path
import pandas as pd
import re
import time
import os 
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from pytube import YouTube
import yt_dlp
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from gpuinfo import GPUInfo

import wave
import contextlib
from transformers import pipeline
import psutil

#import Recroding video & audio
import os
import subprocess
from pynput import keyboard

# Import audio catch
import sounddevice as sd
from scipy.io.wavfile import write

import csv

# Import the audio
import pyaudio

current_key = None

# function to start recording
def on_key_press(key):
    global current_key
    if key == keyboard.Key.esc:
        exit()
    current_key = key

def on_key_release(key):
    global current_key
    current_key = None

def start_recording(output_filename):
    command = ['ffmpeg', '-f', 'avfoundation', '-video_size', '1280x720', '-framerate', '30', '-i', '0:0', output_filename]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)

def stop_recording(process):
    if process:
        process.communicate(input=b'q')

whisper_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}

source_language_list = [key[0] for key in source_languages.items()]

MODEL_NAME = "vumichien/whisper-medium-jp"
lang = "ja"

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)
os.makedirs('output', exist_ok=True)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload

    text = pipe(file)["text"]

    return warn_output + text

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


def speech_to_text(audio_file, selected_source_lang, whisper_model, num_speakers):
    """
    # Transcribe youtube link using OpenAI Whisper
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """
    
    # model = whisper.load_model(whisper_model)
    # model = WhisperModel(whisper_model, device="cuda", compute_type="int8_float16")

    model = WhisperModel(whisper_model, compute_type="int8")
    print("Whisper Model is set up")
    time_start = time.time()
    if(audio_file == None):
        raise ValueError("Error no audio input")
    print(audio_file)

    try:
        # Get duration
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")
        
        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file, **transcribe_options)

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
        print("transcribe audio done with fast whisper")
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
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        if num_speakers == 0:
        # Find the best number of speakers
            try:
                score_num_speakers = {}

                for num_speakers in range(2, 10+1): # simplely chaneg the range will not Work, we need to use try. and whenever the issue popped up, set it as one speaker.
                    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                    score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                    score_num_speakers[num_speakers] = score
                best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
                print(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
            except:
                best_num_speaker = 1
        else:
            best_num_speaker = num_speakers

        if best_num_speaker == 1:
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER ' + str(i + 1)
        else:
            try:  # if the best_number_speaker is greater than 1, but have only 1 person talk for this part of audio
                # Assign speaker label  
                clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
                labels = clustering.labels_
                print("Is over the clustering")
                for i in range(len(segments)):
                    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
            except:
                print("label wrong -_-") 
                for i in range(len(segments)):
                    segments[i]["speaker"] = 'SPEAKER ' + str(i + 1)

        txtFilename = "/Users/chen/Desktop/robot/robot_research/llama2/llama.cpp/in_output/input.txt"
        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''        
            text += segment["text"] + ' '
            with open(txtFilename, 'a') as file:
                        file.write(text)
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
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
        save_path = "output/transcript_result.csv"
        df_results = pd.DataFrame(objects)
        df_results.to_csv(save_path,mode = 'a')
        print(df_results)
        #print(system_info)
        return df_results, system_info, save_path
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)

def main():
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000
    count = 1
    min_volume_threshold = 2000  # Minimum volume to start recording
    silence_duration_threshold = 2  # Duration of silence before stopping (in seconds)
    while True:
        wave_output_filename = f'recording/test{count}.wav'
        p = pyaudio.PyAudio()
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
        
        print("Listening! Timing starts now.")

        frames = []
        is_recording = False
        continue_recording = True
        below_threshold = False

        time_frame_count = 0
        silence_start_time = 0

        while continue_recording:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            max_volume = np.max(audio_data)

            if max_volume > min_volume_threshold and not is_recording:
                is_recording = True
                silence_start_time = time_frame_count
                print("Recording started.")

            if is_recording:
                if max_volume < min_volume_threshold:
                    if not below_threshold:
                        below_threshold = True
                        silence_start_time = time_frame_count
                        print("Volume has decreased, starting silence timer.")
                else:
                    below_threshold = False
                    silence_start_time = time_frame_count

                if time_frame_count > silence_start_time + silence_duration_threshold * sample_rate / chunk_size:
                    if below_threshold:
                        continue_recording = False
                        print("Silence detected, stopping recording.")
                    else:
                        below_threshold = False

            time_frame_count += 1
            if time_frame_count > sample_rate * 60 / chunk_size:  # Timeout after 1 minute
                continue_recording = False
                print("Timeout, stopping recording.")

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(wave_output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        count += 1
        speech_to_text(wave_output_filename, "en", "medium", 0)

if __name__ == "__main__":
    main()
