from ultralytics import YOLO
from PIL import Image
import cv2
import time
import os
from insightface.app.face_analysis import FaceAnalysis
from faceRecognition import FaceRecognition

if __name__ == '__main__':
    # Load Yolo Module
    model = YOLO('yolov8s.pt')

    # Load the insightface
    face_model = FaceAnalysis(name='buffalo_l', providers='CPUExecutionProvider')
    # use first core GPU to Run it, negative number for using CPU
    face_model.prepare(ctx_id=0)

    cap = cv2.VideoCapture(0)

    # Folder to store the image for people
    path = "/Users/chen/Desktop/robot/robot_research/Visual/imageCrop"
    if not os.path.exists(path):
        os.mkdir(path)

    while cap.isOpened():

        success, frame = cap.read()

        if success:

            results = model.track(frame, persist=True)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for i, box in enumerate(boxes):
                    r = box.xyxy[0].astype(int)
                    crop = frame[r[1]:r[3], r[0]:r[2]]
                    filename = str(i) + ".jpg"
                    # get the crop image when only detect person
                    if box.conf[0] > 0.5 and int(box.cls) == 0:
                        filename = os.path.join(path, filename)
                        face_dets = face_model.get(crop)
                        rimg = face_model.draw_on(crop, face_dets)
                        cv2.imwrite(filename, rimg)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
