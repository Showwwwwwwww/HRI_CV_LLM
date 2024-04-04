import sys
import numpy as np
import os
import cv2
import time
import requests
import io
import base64
from PIL import Image
from ultralytics import YOLO
from Visual.detection import FaceRecognition
from Whisper_speaker_diarization.whisper import Whisper
import torch


class Client:

    def __init__(self, address='http://localhost:5000', device=0, **kwargs):
        self.address = address
        self.robot_actions = {
            "say": self.say
        }
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.whisper = Whisper(whisper_model="large-v2", gpu_id=self.device)
        print('Whisper initialized')
        self.face_recognition = FaceRecognition(face_db='./database/face_db', gpu_id=self.device)
        print('face_recognition initialized')
        self.yolo = YOLO('yolov8s.pt')
        self.yolo.to(device=self.device)
        print('YOLO initialized')

    # -------------------------------------------------------------------------------------------------------------------
    # Robot behavior ###################################################################################################
    # -------------------------------------------------------------------------------------------------------------------
    def communicate_behavior(self):
        while True:
            # Module Work --> Send all the transcript to llama
            prompt, detected_person = self.process_image()
            if prompt is not None:
                with open("pipe_py_to_cpp", "w") as pipeOut:
                    print("Sending to C++ from Vision:", prompt)
                    pipeOut.write(prompt + "\n")
                    pipeOut.flush()  # Ensure the message will send successfully
            transcript = self.process_audio(csv_path='./../output/transcript/transcript_result.csv',
                                            detected_person=detected_person)
            if transcript is not None:
                with open("pipe_py_to_cpp", "w") as pipeOut:
                    print("Sending to C++ from Whisper:", transcript)
                    pipeOut.write(transcript + "\n")
                    pipeOut.flush()  # Ensure the message will send successfully

            # Receive the Response from llama
            with open("pipe_cpp_to_py", "r") as pipeIn: # Waiting and reading the response from cpp(llama)
                response = pipeIn.readline().strip()
                print("Received from C++:", response)

            # Send the response to Robot
            if len(response) > 0:
                self.say(response)
            else:
                print("Read No response from llama")

    # -------------------------------------------------------------------------------------------------------------------
    # Robot controls ###################################################################################################
    # -------------------------------------------------------------------------------------------------------------------
    def get_image(self, show=False, save=False, path=None, save_name=None):
        headers = {'content-type': "/image/send_image"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        j = response.json()
        img = np.array(Image.open(io.BytesIO(base64.b64decode(j['img']))))[:, :, ::-1]  # Convert from BGR to RGB
        if show:  # Don't use if running remotely
            cv2.imshow("Pepper_Image", img)
            cv2.waitKey(1)
        if save:
            cv2.imwrite(f"images/{save_name}.png" if path is None else os.path.join(path, save_name), img)
        return img

    def process_image(self, image=None, show=False, save=False, path=None, save_name=None):
        """
        process the image and make prediction. Name, age and gender...
        :param image:
        :return:
        """
        if image is None:
            image = self.get_image()
        results = self.yolo.track(image, conf=0.5, persist=True, tracker='bytetrack.yaml', verbose=False)
        detected_person, faces, prompt = self.face_recognition.recognition(image)
        if show:
            frame = self.face_recognition.draw_on_with_name(image, faces, detected_person)
            cv2.imshow("InsightFace Inference", frame)
        # Send message via pipe, only if the prompt is not 0
        if len(prompt) > 0:
            return prompt, detected_person
        print(f'Prompt is nothing : {prompt}')
        return

    def get_audio(self):
        """
        Get audio file from the sever
        :return:
        """
        # Send the request for get to get the latest audio file
        response = requests.get(self.address)
        # check the response status code
        if response.status_code == 200:
            # 从 Content-Disposition header to get the file name
            # When Sever send audio file to client, the stucture
            # looks like  Content-Disposition: attachment; filename="example.wav"
            content_disposition = response.headers.get('Content-Disposition')
            filename = "latest_recording.wav"  # default file name
            if content_disposition:
                # try to get the file name from content disposition
                filename = content_disposition.split("filename=")[-1].strip("\"'")

            # save the file locally
            with open(filename, 'wb') as audio_file:
                audio_file.write(response.content)
            print(f"Audio file '{filename}' has been downloaded successfully.")
            return filename
        else:
            print(f"Failed to download the audio file. Server responded with status code: {response.status_code}")
            return

    def process_audio(self, csv_path, lang='en', detected_person=None):
        """
        Whisper process audio file and pass the transcription result to llama visa pipe
        :return:
        """
        filename = self.get_audio()
        if filename is not None:
            transcript = self.whisper.speech_to_text(filename, csv_path, lang, detected_person)
            # Send message via pipe, only if the prompt is not 0
            if len(transcript) > 0:
                return transcript
            return

    def say(self, word, verbose=False):
        headers = {'content-type': "/voice/say"}
        response = requests.post(self.address + headers["content-type"], data=word)
        if verbose ^ self.verbose:  # XOR
            print(f"say(word={word})")

    def receive_text(self):
        """
        receive the text from llama and return this text
        :return: String: the response from llama
        """
        with open("pipe_cpp_to_py", "r") as pipeIn:
            response = pipeIn.readline().strip()
            print("Received from C++:", response)
        response = response if len(response) > 0 else None
        return response

    def send_text(self):
        """
        Send text to sever to make pepper Say
        :return:
        """
        word = self.receive_text()
        if word is not None:
            self.say(word)

    def shutdown(self, verbose=False):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        if verbose ^ self.verbose:
            print("shutdown()")
