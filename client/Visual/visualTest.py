from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('yolov8s.pt')

results = model.track(source="0",show = True,conf = 0.3)








