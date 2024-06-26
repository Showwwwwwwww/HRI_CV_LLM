import os
import cv2
import insightface
import numpy as np
import torch
from ultralytics import YOLO
#from utlis import feature_compare,generate_conversation_prompt
from sklearn import preprocessing
import json
from scipy.optimize import linear_sum_assignment

class FaceRecognition2:
    def __init__(self, gpu_id=0, face_db='./../database/face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640),people_info_path = './database/people_info/people_info.json'):
        """
        Face Recognition Tool
        :param gpu_id: Positive number represent the ID for GPU, negative number is for using CPU
        :param face_db: Database for face
        :param threshold: Threshold for compare two face
        :param det_thresh: Detect threshold
        :param det_size: Detect image size
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.people_info_path = people_info_path
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size
        # Loading the model
        self.model = insightface.app.FaceAnalysis(name='buffalo_l',
                                                  providers=['CUDAExecutionProvider' if gpu_id >= 0 else 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # facial feature
        self.faces_embedding = list()

        self.gallery = {}
        # loading the face in db
        self.load_faces(self.face_db)
        self.unknownCount = 0

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
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                # username is the file name
                user_info = file.split(".")[0].split("_")
                user_name = user_info[0]
                age = user_info[1]
                sex = user_info[2] # M or F
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                
                self.gallery[user_name] = {
                            "feature": embedding,  # Store the numpy feature
                            "sex": sex,
                            "age": age,
                        }
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding,
                    "sex": sex,
                    "age": age,
                    'flag': True
                })
        # # write data into json file
        # with open(self.people_info_path, 'w') as json_file:
        #     json.dump(self.faces_embedding, json_file, indent=4)


    def match_new_person(self,face, threshold=0.5):
        embedding = np.array(face.embedding).reshape((1, -1))
        # Get the embedding for this face/person
        new_features = preprocessing.normalize(embedding)
        gallery_names = list(self.gallery.keys())
        gallery_features = np.array([self.gallery[name]["feature"].flatten() for name in gallery_names])
        
        # Calculate the euclidean_distance between the new image and the gallery
        cost_matrix = np.linalg.norm(gallery_features - new_features, axis=1) # Do the multiplication 
        
        # Cosian Similiarity
        cost_matrix = np.dot(gallery_features, new_features.T).flatten() # Larger means more similiar 
        
        print(f'cost_matrix: {cost_matrix}')
        # Hungarian match
        row_ind, col_ind = linear_sum_assignment(cost_matrix.reshape(1, -1))
        # It will return the Hungraian distance for each feature and 
        matched_index = col_ind[0]
        #
        # Check the minimum distance is less than the threshold or not
        
        if cost_matrix[matched_index] > threshold: 
            return gallery_names[matched_index]
        else:
            self.gallery['new_friend'] = {
                 "feature": new_features,  # Store the numpy feature
                "sex": face.sex,
                "age": face.age
            }
            return cost_matrix[matched_index] # When it is None, we identify it is as New Person Coming, And we need to add it to Gallery
        


