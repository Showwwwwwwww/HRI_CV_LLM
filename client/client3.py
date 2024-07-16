import numpy as np
import os
import cv2
import requests
import io
import base64
from PIL import Image
import torch
import traceback
import time
import json
import threading
from llmControl import llm
from Visual.detection2 import FaceRecognition2
from Whisper_speaker_diarization.whisper import Whisper

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Client:

    def __init__(self, address='http://localhost:5001', device=1, **kwargs):
        self.address = address
        self.vertical_ratio = None
        self.horizontal_ratio = None
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info('Using device: {}'.format(self.device))

        self.whisper = Whisper(whisper_model="large-v2", gpu_id=self.device)
        logger.info('Whisper initialized')
        self.face_recognition = FaceRecognition2(face_db='./database/face_db', gpu_id=self.device)
        logger.info('Face Module initialized')
        self.llm = llm()
        logger.info('LLM deployed successfully')

        self.interact_person = None
        self.audio_count = 0
        self.json_path = self.create_new_json_file('./output')
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def communicate_behavior3(self):
        rounds_data = {}
        dic_count = 0
        self.clear_audio_files()
        vision_thread = threading.Thread(target=self.vision_thread)
        vision_thread.start()
        try:
            while True:
                round_info = self.detect_person_and_record_audio()
                if round_info:
                    with self.lock:
                        rounds_data[f"Round{dic_count}"] = round_info
                        dic_count += 1
        except KeyboardInterrupt:
            logger.info("Process interrupted by user.")
            self.cleanup(vision_thread, rounds_data, dic_count)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            self.cleanup(vision_thread, rounds_data, dic_count, error=e)

    def detect_person_and_record_audio(self):
        round_info = {}
        while True:
            with self.lock: # Lock the thread to avoid the conflict
                interactPerson = self.interact_person 
                #print(f'Interacted person is {interactPerson}')
            sound = self.get_sound()
            if interactPerson and sound > 300:
                round_info["Detected person"] = interactPerson
                break
        self.set_eyes_red()
        #self.llm.initialize_llm_talk(interactPerson)
        result = self.process_audio(csv_path='./../output/transcript/transcript_result.csv', detected_person=interactPerson)
        self.set_eyes_blue()
        if result is not None:
            transcript = result[0]
            duration = result[1]
            diff_time = result[2:]
            transcript_info = {
                "Transcript": transcript,
                "Audio Duration": duration,
                "Processing Time": diff_time
            }
            round_info["Audio Info"] = transcript_info
            start_time = time.time()
            response = self.llm.talkTollm(interactPerson, transcript)
            response_time = time.time() - start_time
            self.say(response)
            round_info["Response Info"] = response
            round_info["Response Time"] = response_time
        else:
            self.say("Sorry I have not heard anything")
        return round_info

    def vision_thread(self):
        self.set_eyes_blue()
        logger.info("Vision thread started")
        x = 0
        while not self.stop_event.is_set():
            frame = self.get_image(save=True, path="./output", save_name="Pepper_Image", show=True)
            detectedPerson, track_id, face = self.face_recognition.process_frame(frame, show=False)
            logger.info(f'detected person: {detectedPerson}')
            self.center_target2(detectedPerson, face, frame.shape)
            if self.face_recognition.person_matching(detectedPerson, track_id):
                print(x)
                x += 1 
                continue
            else: # Find the mismatch, we do two more comparsion 
                flag = True
                check_id = track_id
                for _ in range(2):
                    frame = self.get_image(save=True, path="./output", save_name="Pepper_Image")
                    check_person, check_id, face = self.face_recognition.process_frame(frame, show=False)
                    if check_person == detectedPerson:
                        continue
                    else:
                        logger.info(f'Check_person: {check_person}, DetectedPerson: {detectedPerson}')
                        flag = False
                        break
                if flag:
                    track_id = check_id
                    # if self.face_recognition.mismatch_name:
                    #     self.face_recognition.save_cropped_image()
                    #with self.lock:
                    self.face_recognition.set_target_id(track_id)
                    self.face_recognition.set_target_name(check_person)
            with self.lock:
                self.interact_person = self.face_recognition.get_target_name()
                logger.info(f' Interact_person is : {self.interact_person}')

    def clear_audio_files(self):
        path = './../server/recordings'
        for filename in os.listdir(path):
            if filename.endswith('.wav') or filename.endswith('.raw'):
                file_path = os.path.join(path, filename)
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")

    def create_new_json_file(self, target_dir):
        files = os.listdir(target_dir)
        json_files = [file for file in files if file.endswith('.json')]
        json_count = len(json_files)
        new_file_name = f'Test_{json_count + 1}.json'
        new_file_path = os.path.join(target_dir, new_file_name)
        initial_data = {"info": "This is a new JSON file."}
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            json.dump(initial_data, new_file, ensure_ascii=False, indent=4)
        logger.info(f"New JSON file created: {new_file_path}")
        return new_file_path

    def center_target2(self, detected_persons, boxes, img_shape, stop_threshold=0.5, vertical_offset=0.5):
        if not detected_persons or boxes is None or len(boxes) == 0:
            #print("No person detected")
            #logger.info("No person detected")
            pass
        else:
            print(f"Person detected: {detected_persons}")
            # face = boxes[0]
            # box = face.bbox.astype(int)
            box = boxes
            box_center = np.array([box[2] / 2 + box[0] / 2, box[1] * (1 - vertical_offset) + box[3] * vertical_offset])
            frame_center = np.array((img_shape[1] / 2, img_shape[0] / 2))
            diff = frame_center - box_center
            horizontal_ratio = diff[0] / img_shape[1]
            vertical_ratio = diff[1] / img_shape[0]
            logger.info(f"Horizontal ratio: {horizontal_ratio}, Vertical ratio: {vertical_ratio}")
            if 0.05 < abs(horizontal_ratio) <= stop_threshold:
                logger.info(f'Head Roated for forward: {str(vertical_ratio * 0.2)}, left: {str(horizontal_ratio * 0.2)}')
                self.rotate_head(forward=vertical_ratio * 0.4, left=horizontal_ratio * 0.2)
                print(f'Head Roated for forward: {str(vertical_ratio * 0.2)}, left: {str(horizontal_ratio * 0.2)}')
            else:
                logger.info(f'Head Not Roated for forward: {str(vertical_ratio)}, left: {str(horizontal_ratio)}')
            

    def save_json(self):
        for person, file_info in self.llm.conversations.items():
            file_path = os.path.join(self.llm.path, f'{person}.json')
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(file_info, file, ensure_ascii=False, indent=4)
            logger.info(f'People {person} conversation stored')

    def get_image(self, show=False, save=False, path=None, save_name=None):
        headers = {'content-type': "/image/send_image"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        j = response.json()
        img = np.array(Image.open(io.BytesIO(base64.b64decode(j['img']))))[:, :, ::-1]  # Convert from BGR to RGB
        if save:
            if save_name is None:
                raise ValueError("save_name must be provided to save the image.")
            if path is None:
                path = "images"
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, f"{save_name}.jpg")
            cv2.imwrite(file_path, img)
            if show:
                cv2.imshow("Pepper_Image", img)
                cv2.waitKey(1)
        return img

    def get_sound(self):
        try:
            headers = {'content-type': "/audio/volume"}
            response = requests.get(self.address + headers["content-type"])
            j = response.json()
            return j['volume']
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while getting the input volume:", e)
            return

    def download_audio(self):
        try:
            headers = {'content-type': "/audio/recording"}
            response = requests.get(self.address + headers["content-type"], stream=True)
            if response.status_code == 200:
                os.makedirs('downloads', exist_ok=True)
                local_path = os.path.join("downloads", "downloaded_audio.wav")
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"File downloaded successfully: {local_path}")
                return local_path
            else:
                logger.error(f"Failed to download the file: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while downloading the file:", e)
            return None

    def get_audio(self):
        try:
            headers = {'content-type': "/audio/recording"}
            response = requests.get(self.address + headers["content-type"], stream=True)
            if response.status_code == 200:
                content_disposition = response.headers.get('Content-Disposition')
                filename = content_disposition.split('filename=')[1].strip("\"'") if content_disposition else "downloaded_audio.wav"
                local_path = os.path.join("downloaded_files", filename)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"File downloaded successfully: {local_path}")
                return local_path
            else:
                logger.error(f"Failed to download the file: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while downloading the file:", e)
            return None

    def process_audio(self, csv_path, lang='en', detected_person=None, llmOnly=False):
        direct_path = f"./../server/recordings/recording{self.audio_count}.wav"
        self.audio_count += 1
        filename = self.get_audio()
        while not filename:
            filename = self.get_audio()
        if filename:
            if os.path.exists(direct_path):
                result = self.whisper.speech_to_text(direct_path, csv_path, lang, detected_person, llmOnly=llmOnly)
                os.remove(direct_path)
                return result
            logger.info(f"Audio Path not exist: {filename}")
        return

    def say(self, word):
        headers = {'content-type': "/voice/say"}
        response = requests.post(self.address + headers["content-type"], data=word)

    def rotate_head(self, forward=0, left=0, speed=0.2, verbose=False):
        logger.info("Rotate Head has been called")
        headers = {'content-type': "/locomotion/rotateHead"}
        response = requests.post(self.address + headers["content-type"] + f"?forward={str(forward)}&left={str(left)}&speed={str(speed)}")
        if verbose:
            logger.info(f"rotate_head(forward={str(forward)}, left={str(left)}, speed={str(speed)})")

    def shutdown(self):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)

    def cleanup(self, vision_thread, rounds_data, dic_count, error=None):
        self.stop_event.set()
        vision_thread.join()
        if error:
            traceback.print_exc()
        self.save_json()
        with open(self.json_path, 'w') as f:
            json.dump(rounds_data, f, indent=4)
        self.shutdown()
        if error:
            raise error
    
    def set_eyes_blue(self):
        headers = {'content-type': "/led/eyes/blue"}
        response = requests.post(self.address + headers["content-type"])
        if response.status_code == 200:
            logger.info("Eyes set to blue")
        else:
            logger.error(f"Failed to set eyes to blue: {response.text}")

    def set_eyes_red(self):
        headers = {'content-type': "/led/eyes/red"}
        response = requests.post(self.address + headers["content-type"])
        if response.status_code == 200:
            logger.info("Eyes set to red")
        else:
            logger.error(f"Failed to set eyes to red: {response.text}")

    def turn_off_eyes(self):
        headers = {'content-type': "/led/eyes/off"}
        response = requests.post(self.address + headers["content-type"])
        if response.status_code == 200:
            logger.info("Eyes turned off")
        else:
            logger.error(f"Failed to turn off eyes: {response.text}")