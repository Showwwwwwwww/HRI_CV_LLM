from sort import *
from edgeai_yolov5.yolo import YoloManager
import numpy as np
import cv2
import os
import sys

if __name__ == "__main__":
    data_dir = os.path.join("edgeai_yolov5", "data", "photos")

    manager = YoloManager()

    frame1 = cv2.imread(os.path.join(data_dir, "frame1.jpg"))
    frame2 = cv2.imread(os.path.join(data_dir, "frame2.jpg"))

    pred1 = manager.predict(frame1)
    pred2 = manager.predict(frame2)

    pred1_box = manager.extract_bounding_box_data(pred1)
    pred2_box = manager.extract_bounding_box_data(pred2)