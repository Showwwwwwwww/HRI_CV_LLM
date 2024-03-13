import cv2
# Import the audio
import pyaudio

import numpy as np

import wave

from ultralytics import YOLO
from Visual.detection import FaceRecognition
from Whisper_speaker_diarization.whisper import Whisper

def main():
    # load SpeechRecognition Module
    whisper = Whisper()
    print("Whisper Initialized")

    # Load vision Module
    yoloModel = YOLO('yolov8s.pt')
    face_r = FaceRecognition()
    print("Vision Module Initialized")

    # Set up the default parameters for recording video, threshold and volume
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000
    count = 1
    min_volume_threshold = 2000  # Minimum volume to start recording
    silence_duration_threshold = 2  # Duration of silence before stopping (in seconds)

    # Use the camera
    cap = cv2.VideoCapture(0)

    while True:


        wave_output_filename = f'test{count}.wav'
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
        #frame_count = 0

        # Store the detected Person list
        detected_person = None
        while continue_recording and cap.isOpened():
            # analyse video frame
            success, frame = cap.read()
            if success:
                results = yoloModel.track(frame, conf=0.5, persist=True, tracker='bytetrack.yaml')
                #if frame_count % 90 == 0:
                detected_person, faces, prompt = face_r.recognition(frame)
                print(f'Here is the prompt{prompt}')
                frame = face_r.draw_on_with_name(frame, faces, detected_person)

                cv2.imshow("InsightFace Inference", frame)
                #frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

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
        whisper.speech_to_text(wave_output_filename, "en", detected_person)


if __name__ == '__main__':
    main()