import numpy as np
import os
import cv2
import requests
import io
import base64
from PIL import Image
from ultralytics import YOLO
from Visual.detection import FaceRecognition
from Whisper_speaker_diarization.whisper import Whisper
import torch
import traceback
import time
import re
class Client:

    def __init__(self, address='http://localhost:5001', device=0, **kwargs):
        self.address = address
        self.vertical_ratio = None
        self.horizontal_ratio = None
        self.last_box = None
        self.current_target = 'unknown'
        self.device = device if torch.cuda.is_available() else 'cpu'
        print('Using device: {}'.format(self.device))
        # Initialize the Whisper,  FaceRecognition, and Yolo models
        #self.whisper = Whisper(gpu_id=self.device) # Use default medium model
        self.whisper = Whisper(whisper_model="large-v2", gpu_id=self.device)
        print('Whisper initialized')
        self.face_recognition = FaceRecognition(face_db='./database/face_db', gpu_id=self.device)
        #self.face_recognition = FaceRecognition(face_db='./database/face_db', gpu_id=0)
        print('face_recognition initialized')
        self.yolo = YOLO('yolov8s.pt')
        self.yolo.to(device=self.device)
        print('YOLO initialized')
        self.audio_flag = ""
        self.previous_response = "None"
        self.audio_count = 0
    # -------------------------------------------------------------------------------------------------------------------
    # Robot behavior ###################################################################################################
    # -------------------------------------------------------------------------------------------------------------------
    def communicate_behavior(self):
        """
        
        """
        self.clear_audio_files()
        try:
            while True:
                # Module Work --> Send all the transcript to llama
                    # -------------> Vision <----------------
                while True: # If have person has detected in the camera, we start to record the conversation
                    img = self.get_image(save=True, path="./output", save_name="Pepper_Image")
                    prompt, detected_person,faces = self.process_image(img)
                    if detected_person and self.sound_exceed_threshold(): # If we detected person and the input volume exceed the threshold, we can start to record the conversation
                        print("Detected person ", detected_person)
                        break
                path_to_cpp = '/home/shuo/robot_research/output/exchange_information/py_to_cpp.txt'
                if len(prompt) > 0: # If the new recognized person has detected, the prompt will be generated
                    print("Prompt ", prompt)
                    with open(path_to_cpp, "w") as f:
                        f.write(prompt)
                    # with open("pipe_py_to_cpp", "w") as pipeOut:
                    #     print("Sending to C++ from Vision:", prompt)
                    #     pipeOut.write(prompt + "\n")
                    #     pipeOut.flush()  # Ensure the message will send successfully
                    
                    # ------------> Whisper <----------------
                transcript = self.process_audio(csv_path='./../output/transcript/transcript_result.csv',
                                                detected_person=detected_person)
                if transcript:
                    with open(path_to_cpp, "w") as f:
                        f.write(transcript)
                    print(transcript)
                    # with open("pipe_py_to_cpp", "w") as pipeOut:
                    #     print("Sending to C++ from Whisper:", transcript)
                    #     pipeOut.write(transcript + "\n")
                    #     pipeOut.flush()  # Ensure the message will send successfully
                else:
                    self.say("Sorry I have not heard anything")


                if transcript or len(prompt) >0:
                    response = ""
                    try:
                        count = 0
                        path_to_py = '/home/shuo/robot_research/output/exchange_information/cpp_to_py.txt'
                        while len(response) == 0:
                            response = self.process_response_llama(path_to_py)
                            if len(response) > 0:
                                print(f'response when broke: {response}')
                                break
                            # with open(path_to_py, "r") as f:
                            #     print("Read from", path_to_py)
                            #     response = f.read()
                            #     if len(response) > 0:
                            #         break
                            # # Receive the Response from llama
                            # with open("pipe_cpp_to_py", "r") as pipeIn: # Waiting and reading the response from cpp(llama)
                            #     response = pipeIn.readline().strip()
                            #     print("Received from C++:", response)
                            time.sleep(1)
                        with open(path_to_py, "w") as f:
                            print("Clear path", path_to_py)
                            pass
                        if response != self.previous_response:
                            self.say(response)
                            #print('response: ', response)
                            self.previous_response = response
                        #print(count)
                        count += 1
                    except Exception as e:
                        print(f"An error occured whic receiving the response from C++{e}")

        except Exception as e:
            #print(e)
            traceback.print_exc()
            self.shutdown()

    def exp_tracking(self):
        frameCount = 0
        while frameCount < 100:
            person_joined = False 
            while not person_joined: # If have person has detected in the camera, we start to record the conversation
                prompt, detected_person,faces = self.process_image()
                person_joined = len(faces) > 0 # Have person face to camera



    # -------------------------------------------------------------------------------------------------------------------
    # Information Process  ###################################################################################################
    # -------------------------------------------------------------------------------------------------------------------
    def center_target(self, box, img_shape, stop_threshold = 0.1, vertical_offset=0.5, detected_Person=False):
        """ Takes in target bounding box data and attempts to center it
        Preconditons:
            1. box must contain data about exactly 1 bounding box

        Params:
            box: 2D array
                Data about one bounding box in the shape of: 1 x 5
                Columns must be in the format of (x1, y1, x2, y2, id)
            img_shape: 1D array
                Shape of the original frame in the format: (height, width, colour channels),
                so passing in original_image.shape will do.
            stop_threshold: float between 0 and 1
                If the difference between box center and frame center over frame resolution is less than this threshold,
                tell the robot to stop rotating, otherwise, the robot will be told to rotate at a rate proportional to
                this ratio.
            vertical_offset: float between 0 and 1
        """
        if len(img_shape)!=3: # Check shape of image
            raise Exception(f"The shape of the image does not equal to 3!")

        if len(box)>1: # Check number of tracks
            # If not 1, then the target is either lost, or went off-screen
            #raise Exception(f"The length of box is {len(box)}, but it should be 1!")
            # self.stop()
            if self.target_person not in detected_Person:
                print("Target Lost")
                #self.target_lost()
            else:
                self.rotate_head_abs()
        elif len(box) == 0:
            # If the length of box is zero, that means Pepper just lost track of the target before it officially
            # declares the target lost. In this window, we can still recover the track by making Pepper move towards
            # wherever the target could've shifted to
            # if self.vertical_ratio is not None and self.horizontal_ratio is not None and self.dl_model.target_id!=0:
            #     self.walkToward(theta=self.horizontal_ratio*1.5)
            pass
        else: # If there's only 1 track, center the camera on them
            # Since there's an extra dimension, we'll take the first element, which is just the single detection
            face  = box[0]
            box = face.bbox.astype(int) # from insight face
            # Following shapes will be (x, y) format
            box_center = np.array([box[2]/2+box[0]/2, box[1]*(1-vertical_offset)+box[3]*vertical_offset])#box[1]/2+box[3]/2])
            frame_center = np.array((img_shape[1]/2, img_shape[0]/2))
            #diff = box_center - frame_center
            diff = frame_center - box_center
            horizontal_ratio = diff[0]/img_shape[1]
            vertical_ratio = diff[1]/img_shape[0]

            # Saves a copy of the last ratio
            self.vertical_ratio = vertical_ratio
            self.horizontal_ratio = horizontal_ratio

            if abs(horizontal_ratio) <= stop_threshold:
                self.approach_target(box, img_shape, vertical_ratio)

    def clear_audio_files(self):
        path = './../server/recordings'
        # 检查目录中的每个文件
        for filename in os.listdir(path):
            # 检查文件扩展名是否是.wav或.raw
            if filename.endswith('.wav') or filename.endswith('.raw'):
                # 构建完整的文件路径
                file_path = os.path.join(path, filename)
                # 删除文件
                os.remove(file_path)
                print(f"Deleted {file_path}")

    def process_response_llama(self, file_path):


        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Search Bob: to "User: " or the end of the txt
        match = re.search(r'Bob: (.*?)(?:User:|$)', content, re.DOTALL)
        if not match:
            #print("No content found after 'Bob: '")
            return ""

        # Get the extract txt
        text = match.group(1)

        # Remove all content in * *, it represents the emotion
        cleaned_text = re.sub(r'\*([^*]+)\*', '', text)

        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text)

        print(cleaned_text)
        # Return the txt, and remove the empty space
        return ' '.join(cleaned_text.strip().split())


    def center_target2(self, detected_persons, boxes, img_shape, stop_threshold = 0.5, vertical_offset=0.5):
        """
         
        :param detected_persons: a List contained the name for the detected persons, 'unknow' or 'name'
        :param boxes: a List contained the bounding box for the corresponding detected persons
        :param img_shape: the shape of the image
        """
        if len(detected_persons) == 0:
            print("No person detected")
            #self.rotate_head_abs()
        else:
            print("Have perosn detected")
            # # Attempt to update current_target at first
            # if self.current_target == 'unknown': # if the current target is unknow, we attep to update it as a person
            #     print("Current target is unknown")
            #     for person in detected_persons:
            #         if person != self.current_target: # Find the first person that is not the current target
            #             print(f"Current target is updated from {self.current_target} to {person}")
            #             self.current_target = person # Update and break the loop
            #             break
            #     self.current_target = 
            
            for i in range(len(detected_persons)):
                #if detected_persons[i] == self.current_target:  # if we found the target person in the image, rotate the head and point to him
                print(f"Target person {self.current_target} detected and tracking")
                face = boxes[i]
                box = face.bbox.astype(int)
                box_center = np.array([box[2]/2+box[0]/2, box[1]*(1-vertical_offset)+box[3]*vertical_offset])#box[1]/2+box[3]/2])
                frame_center = np.array((img_shape[1]/2, img_shape[0]/2))
                #diff = box_center - frame_center
                # print(f"box_center: {box_center}, frame_center: {frame_center}")

                diff = frame_center - box_center
                horizontal_ratio = diff[0]/img_shape[1]
                vertical_ratio = diff[1]/img_shape[0]
                print(f"Horizontal ratio: {horizontal_ratio}, Vertical ratio: {vertical_ratio}")
                # Saves a copy of the last ratio
                self.vertical_ratio = vertical_ratio
                self.horizontal_ratio = horizontal_ratio
                # print(f"Horizontal ratio: {horizontal_ratio}, Vertical ratio: {vertical_ratio}")

                if abs(horizontal_ratio) <= stop_threshold:
                    print("Rotate Head")
                    self.rotate_head(forward=-(vertical_ratio*0.3),left= horizontal_ratio*0.4)
                    #self.approach_target(box, img_shape, vertical_ratio,horizontal_ratio)
                else:
                    print("Not Rotate Head")
                break
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
            if save_name is None:
                raise ValueError("save_name must be provided to save the image.")
            
            # Ensure the directory exists
            if path is None:
                path = "images"
            if not os.path.exists(path):
                os.makedirs(path)
            
            # Construct the full file path with the correct extension
            file_path = os.path.join(path, f"{save_name}.jpg")
            
            # Attempt to save the image
            cv2.imwrite(file_path, img)
        return img

    def sound_exceed_threshold(self):
        try:
            headers = {'content-type': "/audio/volume"}
            response = requests.get(self.address + headers["content-type"])
            j = response.json()
            print(f'this is current volume: {j["volume"]}')
            return j['volume'] > 1000
        except requests.exceptions.RequestException as e:
            print("An error occurred while getting the input volume:", e)
            return 
        
    def download_audio(self):
        """
        Download an audio file from the given URL and return the local file path
        return: str: the local file path of the downloaded audio file
        """
        try:
            headers = {'content-type': "/audio/recording"}
            response = requests.get(self.address + headers["content-type"], stream=True)
            if response.status_code == 200:
                # Ensure the directory exists
                os.makedirs('downloads', exist_ok=True)
                # Specify the local path where the file should be saved
                local_path = os.path.join("downloads", "downloaded_audio.wav")
                # Write the response content to a file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"File downloaded successfully: {local_path}")
                return local_path
            else:
                print("Failed to download the file:", response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            print("An error occurred while downloading the file:", e)
            return None
        
    def get_audio(self):
        """
        Download an audio file from the given URL and return the local file path
        return: str: the local file path of the downloaded audio file
        """
        try:
            headers = {'content-type': "/audio/recording"}
            response = requests.get(self.address +headers["content-type"], stream=True)
            if response.status_code == 200:
                # Get the filename from Content-Disposition header
                content_disposition = response.headers.get('Content-Disposition')
                filename = content_disposition.split('filename=')[1].strip("\"'") if content_disposition else "downloaded_audio.wav"
                
                # Specify the local path where the file should be saved
                local_path = os.path.join("downloaded_files", filename)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Write the response content to a file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"File downloaded successfully: {local_path}")
                return local_path
            else:
                print("Failed to download the file:", response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            print("An error occurred while downloading the file:", e)
            return None
        
    def process_image(self, image=None, show=False, save=False, path=None, save_name=None):
        """
        process the image and make prediction. Name, age and gender...
        :param image:
        :return:
        """
        try:
            if image is None:
                image = self.get_image()
            #print("Image shape:", image.shape)
            results = self.yolo.track(image, conf=0.5, persist=True, tracker='bytetrack.yaml', 
                                    verbose=False)
            detected_person, faces, prompt = self.face_recognition.recognition(image)
            #self.center_target(faces, image.shape)
            #print(f"len of faces: {len(faces)}")
            self.center_target2(detected_person, faces, image.shape)

            if show:
                frame = self.face_recognition.draw_on_with_name(image, faces, detected_person)
                # cv2.imshow("InsightFace Inference", frame)
            # Send message via pipe, only if the prompt is not 0

            return prompt, detected_person, faces # faces represent if this conversation have faces
            #print(f'Prompt is nothing : {prompt}')
        except Exception as e:
            #print(f"An error occurred while processing the image: {e}")
            return
        
    
    def process_audio(self, csv_path, lang='en', detected_person=None):
        """
        Whisper process audio file and pass the transcription result to llama visa pipe
        :return:
        """
        direct_path = f"./../server/recordings/recording{self.audio_count}.wav"
        self.audio_count += 1
        filename = self.get_audio()
        while not filename:
            filename = self.get_audio()
        if filename:
            if os.path.exists(direct_path):
                transcript = self.whisper.speech_to_text(direct_path, csv_path, lang, detected_person)
                #os.remove(direct_path)
                # return the transcript
                return transcript if transcript else None
            print("Path not exist")
        return

    def say(self, word):
        headers = {'content-type': "/voice/say"}
        response = requests.post(self.address + headers["content-type"], data=word)

    def rotate_head(self, forward=0, left=0, speed=0.2, verbose=False):
        print("Rotate Head has been call")
        headers = {'content-type': "/locomotion/rotateHead"}
        response = requests.post(self.address + headers[
            "content-type"] + f"?forward={str(forward)}&left={str(left)}&speed={str(speed)}")
        # if verbose ^ self.verbose:
        #     print(f"rotate_head(forward={str(forward)}, left={str(left)}, speed={str(speed)})")

    def rotate_head_abs(self, forward=0, left=0, speed=0.2, verbose=False):
        headers = {'content-type': "/locomotion/rotateHeadAbs"}
        response = requests.post(self.address + headers[
            "content-type"] + f"?forward={str(forward)}&left={str(left)}&speed={str(speed)}")
        # if verbose ^ self.verbose:
        #     print(f"rotate_head_abs(forward={str(forward)}, left={str(left)}, speed={str(speed)})")

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

    def shutdown(self):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)

