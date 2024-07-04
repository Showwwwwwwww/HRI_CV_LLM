import numpy as np
import os
import cv2
import requests
import io
import base64
from PIL import Image
# from ultralytics import YOLO
from Visual.detection2 import FaceRecognition2
#from Visual.detection import FaceRecognition
from Whisper_speaker_diarization.whisper import Whisper
import torch
import traceback
import time
import re
import json
import threading
from llmControl import llm



class Client:

    def __init__(self, address='http://localhost:5001', device=1, **kwargs):
        self.address = address
        self.vertical_ratio = None
        self.horizontal_ratio = None
        self.device = device if torch.cuda.is_available() else 'cpu'
        print('Using device: {}'.format(self.device))

        
        #self.whisper = Whisper(gpu_id=self.device) # Use default medium model
        self.whisper = Whisper(whisper_model="large-v2", gpu_id=self.device)
        print('Whisper initialized')
        self.face_recognition = FaceRecognition2(face_db='./database/face_db', gpu_id=self.device)
        print('Face Module initialized')
        self.llm = llm()
        print('LLM delpoy Success')

        self.audio_count = 0
        self.json_path = self.create_new_json_file('./output')
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------------------------------------------------
    # Robot behavior ###################################################################################################
    # -------------------------------------------------------------------------------------------------------------------
    def communicate_behavior3(self):
        rounds_data = {}
        dic_count = 0
        self.clear_audio_files()
        vision_thread = threading.Thread(target=self.vision_thread)
        vision_thread.start()
        prompt = None
        try:
            while True:
                round_info = {}
                while True:
                    interactPerson = self.face_recognition.get_target_name()
                    if interactPerson and self.sound_exceed_threshold():
                        round_info["Detected person"] = interactPerson
                        round_info["Prompt"] = self.prompt or None
                        break
                self.say("Start recording audio")
                result = self.process_audio(csv_path='./../output/transcript/transcript_result.csv',
                                            detected_person=interactPerson)
                self.say("Audio recording finished")
                transcript = result[0]
                duration = result[1]
                diff_time = result[2:]
                transcript_info = {}
                transcript_info["Transcript"] = transcript
                transcript_info["Audio Duration"] = duration
                transcript_info["Processing Time"] = diff_time
                round_info["Audio Info"] = transcript_info
                if transcript:
                    start_time = time.time()
                    response = self.llm.talkTollm(interactPerson,transcript)
                    response_time = time.time() - start_time
                    print(f'{interactPerson} say : {transcript}')
                    self.say(response)
                    print(f'Pepper say: {response}')
                    round_info["Response Info"] = response
                    round_info["Response Time"] = response_time
                    #print(transcript)
                else:
                    self.say("Sorry I have not heard anything")
                round_info['Eye Contact Rate'] = self.eyeContact / self.frameCount
                rounds_data[f"Round{dic_count}"] = round_info # We save this the information in this round to the big frame data
                # with open(self.json_path, 'w') as f:
                #     json.dump(rounds_data, f, indent=4) # Save it each time
                dic_count+=1
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            self.stop_event.set()
            vision_thread.join()
            #round_info['Eye Contact Rate'] = self.eyeContact / self.frameCount
            rounds_data[f"Round{dic_count}"] = round_info
            with open(self.json_path, 'w') as f:
                json.dump(rounds_data, f, indent=4)
            self.shutdown()

        except Exception as e:
            print(f"An error occurred: {e}")
            self.stop_event.set()
            vision_thread.join()
            traceback.print_exc()
            with open(self.json_path, 'w') as f:
                json.dump(rounds_data, f, indent=4)
            self.shutdown()



    # -------------------------------------------------------------------------------------------------------------------
    # Information Process  ###################################################################################################
    # -------------------------------------------------------------------------------------------------------------------
    def vision_thread(self):
        """
        Functionality: 1. ReID, 2. Detect if person currently in the frame 
        """
        while not self.stop_event.is_set(): 
            #self.frameCount += 1
            print("Vision thread running")
            frame = self.get_image(save=True, path="./output", save_name="Pepper_Image")
            detecedPerson,track_id, face = self.face_recognition.process_frame(frame,show=True)
            if face: # If there is face return, we return the position and reset the 
                self.center_target(face, frame.shape)
            if self.face_recognition.person_matching(detecedPerson,track_id): # Current person = Previous, but ID not matched 
                continue
            else: # Person changed, or no person in the frame 
                flag = True
                for _ in range(2): 
                    frame = self.get_image(save=True, path="./output", save_name="Pepper_Image")
                    check_person,track_id,face = self.face_recognition.process_frame(frame,show=True)
                    if check_person == detecedPerson:
                        continue
                    else:
                        print(f'Check_person: {check_person}, DetectedPerson: {detecedPerson}')
                        flag = False
                        break
                if flag: # if success, update the name 
                    self.face_recognition.set_target_id(track_id) # People not changed, but the id changed 
                    self.face_recognition.set_target_name(check_person) # Update the current people in the framem, either None or the person name 
            
            time.sleep(1)  # Adjust the sleep time as needed

    def clear_audio_files(self):
        """
        Clear all the audio files in the server
        """
        path = './../server/recordings'

        for filename in os.listdir(path):

            if filename.endswith('.wav') or filename.endswith('.raw'):

                file_path = os.path.join(path, filename)

                os.remove(file_path)
                print(f"Deleted {file_path}")

    def create_new_json_file(self,target_dir):
        """
        Create a new JSON file in the target directory to stores the conversation information in this experiment
        """

        files = os.listdir(target_dir)
        json_files = [file for file in files if file.endswith('.json')]
        json_count = len(json_files)
        new_file_name = f'Test_{json_count + 1}.json'
        new_file_path = os.path.join(target_dir, new_file_name)
        initial_data = {"info": "This is a new JSON file."}
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            json.dump(initial_data, new_file, ensure_ascii=False, indent=4)
        
        print(f"New JSON file created: {new_file_path}")
        return new_file_path


    def center_target(self, face, img_shape, stop_threshold = 0.5, vertical_offset=0.5):
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
        print(f"Horizontal ratio: {horizontal_ratio}, Vertical ratio: {vertical_ratio}")

        if abs(horizontal_ratio) >= stop_threshold or abs(vertical_ratio) >= vertical_offset:
            print("Rotate Head")
            self.rotate_head(forward=-(vertical_ratio*0.3),left= horizontal_ratio*0.4)
            #self.approach_target(box, img_shape, vertical_ratio,horizontal_ratio)
        else:
            print("Not Rotate Head")

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
            print(f"file has been saved at {file_path}")
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
        

    
    def process_audio(self, csv_path, lang='en', detected_person=None, llmOnly = False):
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
                #result = self.whisper.speech_to_text(direct_path, csv_path, lang, detected_person,llmOnly=llmOnly)
                #os.remove(direct_path)
                # return the transcript
                #return transcript if transcript else None
                return ''
                #return result 
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

    def shutdown(self):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)


