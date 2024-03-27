import os
import cv2
import insightface
import numpy as np
import torch
from ultralytics import YOLO
from utlis import feature_compare,generate_conversation_prompt
#from Visual.utlis import feature_compare,generate_conversation_prompt
from sklearn import preprocessing

class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='./../database/face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
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
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size

        # Loading the model
        self.model = insightface.app.FaceAnalysis(name='buffalo_l',
                                                  providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # facial feature
        self.faces_embedding = list()
        # loading the face in db
        self.load_faces(self.face_db)


    # loading the face in db with feature
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                # username is the file name
                user_name = file.split(".")[0]
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding,
                    "sex": "Unknown",
                    "age": "Unknown",
                    'flag': True
                })

    def recognition(self, image):
        faces = self.model.get(image)
        detectedPerson = list()
        prompt = ""
        for face in faces:
            # Start facial detection
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            user_name = "unknown"
            for com_face in self.faces_embedding:
                r = feature_compare(embedding, com_face["feature"], self.threshold)
                # If this is the person who stored in the db
                if r:
                    user_name = com_face["user_name"]
                    # If the flag is true, we only do it once
                    if com_face["flag"]:
                        com_face["age"] = face.age
                        com_face["sex"] = face.sex
                        com_face["flag"] = False  #flag==False means we have already generated the prompt to this person
                        generate_conversation_prompt(com_face)
            detectedPerson.append(user_name)

        return detectedPerson, faces, prompt

    def draw_on_with_name(self, img, faces, names):
        # Add the name on the pic
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                sex = face.sex
                age = face.age
                for one_face in self.faces_embedding:
                    if one_face["user_name"] == names[i]:
                        sex = one_face["sex"]
                        age = one_face["age"]
                cv2.putText(dimg, '%s,%s,%d' % (names[i], sex, age), (box[0] - 1, box[1] - 4),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 1)

        return dimg


if __name__ == '__main__':
    device = 2 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8s.pt')
    model.to(device)
    face_r = FaceRecognition(face_db='face_db')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        count = 0
        if success:

            results = model.track(frame, conf=0.5, persist=True, tracker='bytetrack.yaml',device=0)
            # annotated_frame = results[0].plot()
            if count % 90 == 0:
                names, faces, prompt = face_r.recognition(frame)
                print(f'Here is the prompt{prompt}')
                frame = face_r.draw_on_with_name(frame, faces, names)
            cv2.imshow("InsightFace Inference", frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()