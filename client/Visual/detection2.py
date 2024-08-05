import os
import cv2
import insightface
import numpy as np
import torch
from ultralytics import YOLO
from torch.nn import functional as F
#from utlis import feature_compare,generate_conversation_prompt
from sklearn import preprocessing
import json
from scipy.optimize import linear_sum_assignment

class FaceRecognition2:
    def __init__(self, cuda='cuda', face_db='./database/face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640),people_info_path = './database/people_info/people_info.json'):
        """
        Face Recognition Tool
        :param gpu_id: Positive number represent the ID for GPU, negative number is for using CPU
        :param face_db: Database for face
        :param threshold: Threshold for compare two face
        :param det_thresh: Detect threshold
        :param det_size: Detect image size
        """
        self.cuda = cuda
        self.face_db = face_db
        self.people_info_path = people_info_path
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size
        # Loading the model
        self.model = insightface.app.FaceAnalysis(name='buffalo_l',
                                                  providers=['CUDAExecutionProvider' if self.cuda == 'cuda' else 'CPUExecutionProvider'])
        # get gpu id
        self.gpu_id = torch.cuda.current_device() if self.cuda == 'cuda' else -1
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        print("Insight Face Initalized")
        
        self.yolo = YOLO('yolov8s.pt') 
        self.yolo.to(device=self.cuda)
        print("Yolo initialized")

        self.gallery = {}
        
        # loading the face in db
        self.load_faces(self.face_db)
        print("Face embedding/info initialized")

        # Initialized the person for detected 
        self.target_id = None 
        self.target_name = False
        
        # For add new person into the gallery 
        #self.mismatch_feature = np.empty((0, 512)) # For decide add the person to the gallary
        self.mismatch_feature = torch.empty((0, 512)).cuda()
        self.mismatch_name = None
        self.mismatch_age = None  # This is the group rather than a specific age
        self.mismatch_sex = None
        self.mismatch_crop = None 
        self.crop_name = None
        self.crop_frame = None # For store the crop information for the person

    # loading the face in db with feature
    def load_faces(self, face_db_path):
        """
        Initialize the  face db, to generate all the embedding for the face in the db. And store the result into a json file.
        face file name is name_age_gender.jpg
        """
        print(f'Loading face_db from: ',os.path.abspath(self.face_db))
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                print(f'root: {root }file: {file}')
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                # username is the file name
                user_info = os.path.splitext(file)[0].split("_")
                user_name = user_info[0]
                age = None
                sex = None
                if len(user_info) == 3:
                    age = user_info[1]
                    sex = user_info[2] # M or F

                face = self.model.get(input_image)[0]
                embedding = F.normalize(torch.from_numpy(face.embedding).unsqueeze(0),dim=1).cuda()
                
                self.gallery[user_name] = {
                            "feature": embedding,  # Store the numpy feature
                            "sex": sex if sex else face.sex,
                            "age": age if age else face.age,
                            "crop": 'None'
                        }
                print(f"Load {user_name} into gallery")
               
                    

    def save_cropped_image(self,name = None):
        # Update crop, mismatch, and mismatch_name
        save_path = os.path.join(self.face_db, f'{name}.jpg')
        
        if name and name == self.mismatch_name:
            self.gallery[self.mismatch_name] = {
                                "feature": self.mismatch_feature,  # Store the numpy feature
                                "sex": self.mismatch_sex,
                                "age": self.mismatch_age,
                                "crop": self.mismatch_crop
                            }
            self.mismatch_feature = np.empty((0, 512))
            self.mismatch_name = None
            self.mismatch_sex = None
            self.mismatch_age = None
            self.mismatch_crop = None
        # Save the cropped image
        #cv2.imwrite(save_path, self.crop_frame)
        print(f'name: {name} crop_name: {self.crop_name}')
        if name == self.crop_name and self.crop_frame is not None and name is not None:
            cv2.imwrite(save_path, self.crop_frame)
        # crop = self.gallery[name]["crop"]
        # if crop != 'None':
        #     cv2.imwrite(save_path, crop)
        #     print(self.gallery.keys())
        #     print(name)
        #     self.gallery[name]["crop"] = 'None'
        return save_path
            
    def set_target_id(self, target_id):
        print(f" The target id is set from {self.target_id} to {target_id}.")
        self.target_id = target_id
    def get_target_id(self):
        return self.target_id
    def set_target_name(self, target_name):
        print(f" The target name is set from {self.target_name} to {target_name}")
        self.target_name = target_name
    def get_target_name(self):
        return self.target_name
    
    def person_matching(self,new_person,track_id):
        if new_person == self.get_target_name():
            if track_id != self.get_target_id():
                print(f' The person matched, whihc the name is {new_person}')
                print(f" The person mismatch with track_id {self.target_id}, which changed to {track_id}")
                self.target_id = track_id
                #self.set_target_id(track_id)
            return True
        else:
            return False
        
    def match_new_person(self,face, threshold=0.2):
        new_features = F.normalize(torch.from_numpy(face.embedding).unsqueeze(0),dim=1).cuda()
        # Get the embedding for this face/person
        gallery_names = list(self.gallery.keys())
        gallery_features = torch.cat([self.gallery[name]["feature"] for name in gallery_names], dim=0).cuda()

        if self.mismatch_feature.shape[0] > 0: # Have Mismatch,
            gallery_features = torch.cat([gallery_features, self.mismatch_feature], dim=0).cuda()
            gallery_names.append(self.mismatch_name)

        # Calculate the euclidean_distance between the new image and the gallery
        #cost_matrix = np.linalg.norm(gallery_features - new_features, axis=1) # Do the multiplication 
        
        # Cosian Similiarity
        # cost_matrix = np.dot(gallery_features, new_features.T).flatten() # Larger means more similiar
        cost_matrix = torch.mm(gallery_features, torch.tensor(new_features).T).flatten().cpu().numpy()
        print(f'cost_matrix: {cost_matrix}')
        # Find the highest similarity
        matched_index = np.argmax(cost_matrix)
        max_similarity = cost_matrix[matched_index]

        # -- Because we are using one to one matching, so we can direcrtly use the cos similarity to compare ---
        # Convert similarities to costs (since Hungarian algorithm works with cost minimization)
        # cost_matrix = 1 - cost_matrix
        # # Hungarian match
        # row_ind, col_ind = linear_sum_assignment(cost_matrix.reshape(1, -1))
        # # It will return the Hungraian distance for each feature and 
        # matched_index = col_ind[0]
        # max_similarity = cost_matrix[matched_index]
        
        # Check the minimum distance is greater than the threshold or not
        if max_similarity > threshold: 
            # Update the matched person's embedding with the new embedding
            matched_name = gallery_names[matched_index]
            if matched_name == self.mismatch_name and face['det_score'] > 0.3:
                self.mismatch_feature = new_features
            else: 
                if face['det_score'] > 0.4:
                    self.gallery[matched_name]["feature"] = new_features
                self.mismatch_feature = torch.empty((0, 512)).cuda()
                self.mismatch_name = None
            return matched_name
        else:
            if not self.mismatch_name and face['det_score'] > 0.2: # No mismatch person, and we found one
                self.mismatch_name = 'UnknownPerson' + str(len(self.gallery))
                self.mismatch_feature = new_features
                self.mismatch_age = face.age
                self.mismatch_sex = face.sex
            else: # Already have a mismatch person but not match
                self.mismatch_feature = torch.empty((0, 512)).cuda()
                self.mismatch_name = None
            return None # When it is None, we identify it is as New Person Coming, And we need to add it to Gallery
        
    def process_frame(self,frame,show = False, new_person = False):
        """
        Analysis the identity of current people in the frame
        Return:
            True: If the person in the frame, and just the mod
            False: Means we need to check if the person is new or not 
        """
        results = self.yolo.track(frame, conf=0.75, persist=True, tracker='bytetrack.yaml', 
                                    verbose=False)
        if show:
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", frame)

        for result in results:
            boxes = result.boxes.cpu().numpy() 
            if result.boxes.id is None: 
                continue
            ids = result.boxes.id.cpu().numpy()
            if self.target_id in ids: # People still in the frame 
                return self.get_target_name(), self.get_target_id()
            # Find all person, fin the box 
            for box, track_id in zip(boxes,ids):  # Find each person in the frame, Assume only one person in the frame 
                if int(box.cls) == 0 and box.conf > 0.7 or track_id == self.get_target_id():
                    r = box.xyxy[0].astype(int) 
                    self.crop_frame = crop = frame[r[1]:r[3], r[0]:r[2]] # Person Crop based on it boudning box 
                    face_info = self.model.get(crop) # Information for ananlysis this face. have only one 
                    if len(face_info) >= 1:# Have person face
                        face = face_info[0] 
                        #print(face['det_score'] )
                        if face['det_score'] < 0.3: # Person face is not clear
                            continue
                    else: # No person face or face is not clear
                        continue
                    print(f' target id : {self.target_id} not in the frame ids: {ids}')
                    storedPerson = self.match_new_person(face,threshold=0.3) # Match the person and  Save the image
                    if face['det_score'] > 0.7:
                        if storedPerson in self.gallery.keys():
                            self.gallery[storedPerson]["crop"] = crop

                        else:
                            self.mismatch_crop = crop
                        self.crop_frame = crop
                        self.crop_name = storedPerson
                    return storedPerson, track_id
        return None,None

    def extractBox(self,frame):
        face_info = self.model.get(frame)
        if len(face_info) >= 1:
            face = face_info[0]
            return face.bbox.astype(int)
        else:
            return None
