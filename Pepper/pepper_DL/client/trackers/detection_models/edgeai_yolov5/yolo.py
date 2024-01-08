import torch
import os
import cv2
import csv
import numpy as np
import pandas as pd
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import colors, plot_one_box

class YoloManager():
    def __init__(self, weights=None, device="cpu", save_txt_tidl=True, image_size=[640,640], kpt_label=True):
        self.device = select_device(device)
        self.half = self.device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA
        if weights is None:
            weights_dir = os.path.join(os.path.dirname(__file__), "weights", 'yoloposes_640_lite.pt')
        # Load model
        self.model = attempt_load(weights_dir, map_location=self.device)  # load model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.kpt_label = kpt_label
        #print(torch.cuda.is_available())
        #print(torch.cuda.device_count())
        #print(torch.cuda.current_device())

        if isinstance(image_size, (list,tuple)):
            assert len(image_size) ==2; "height and width of image has to be specified"
            image_size[0] = check_img_size(image_size[0], s=self.stride)
            image_size[1] = check_img_size(image_size[1], s=self.stride)
        else:
            image_size = check_img_size(image_size, s=self.stride)  # check img_size
        self.image_size = image_size
        if self.half:
            self.model.half()


    def preprocess_frame(self, frame):
        # Preprocesses a frame that's in numpy array format to be compatible with self.model
        # Padded resize

        img = letterbox(frame, self.image_size, stride=self.stride, auto=False)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #print(img.shape)
        return img

    def extract_bounding_box_data(self, prediction):
        """ Takes in prediction and outputs bounding box information for all detections
        Params:
            prediction - output of self.predict

        Returns:
            a Tensor in the shape of:
                [[x1, y1, x2, y2, conf]]
        """
        return prediction[:, :5]

    def extract_keypoint_data(self, prediction):
        # Extracts prediction keypoints and reshapes them into:
        #   (number of detections, 17 key points, 3 features consisting of x, y, confidence)
        return prediction[:, 6:].reshape(prediction.shape[0], 17,3)

    def extract_bounding_box_and_keypoint(self, prediction):
        return self.extract_bounding_box_data(prediction), self.extract_keypoint_data(prediction)

    def predict(self, frame, augment=False, conf_thres=0.25, classes=None, iou_thres=0.45, agnostic_nms=False, preprocess=True, scale_to_original=True):
        if preprocess:
            original_shape = frame.shape
            frame = self.preprocess_frame(frame)
        pred = self.model(frame, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=self.kpt_label)[0]
        if scale_to_original:
            self.scale_to_original(pred, original_shape)

        # Torch tensors are different depending on whether they're using GPU or CPU, so we'll have to made some changes
        if self.device.type != "cpu":
            pred = pred.detach().cpu()

        return pred

    def scale_to_original(self, prediction, original_shape):
        scale_coords(self.image_size, prediction[:, :4], original_shape, kpt_label=False)
        scale_coords(self.image_size, prediction[:, 6:], original_shape, kpt_label=self.kpt_label, step=3)

    def draw(self, prediction, img, show=True):
        # for det_index, (*xyxy, conf, cls) in enumerate(reversed(prediction[:,:6])):
        for det_index, (*xyxy, conf, cls) in enumerate(prediction[:,:6]):
            plot_one_box(xyxy, img, label=(f'{self.names[int(cls)]} {conf:.2f}'), color=colors(int(cls),True), line_thickness=2, kpt_label=self.kpt_label, kpts=prediction[det_index, 6:], steps=3, orig_shape=img.shape[:2])
        if show:
            cv2.imshow("Image", img)
            cv2.waitKey(0)

    def draw_and_save(self, in_dir, out_dir=None, rescale=None, conf_thres=0.25, iou_thres=0.45, save_as="jpg"):
        """ Runs sort on every single frame in in_dir and saves it to out_dir
        Params:
          in_dir - directory of images we wish to run tracking on
          out_dir - directory to save images with tracked bounding boxes. If None, makes a new file in the parent dir with
            the same name as in_dir but with added '_sort' in the end
          rescale - a tuple of length 2 where the first element details the height in pixels and the second element details
            the width in pixels. Used to scale input images before feeding to Yolo
          save_as - a string as one of:
                'jpg' - saves each frame as a .jpg file with drawn bounding boxes and skeleton points
                'csv' - saves bounding box information in a .csv file that starts off with a number indicating the frame
                        it's from followed by predicted bounding box points. Used by SORT.
                'both' - does two of the above
        Requires: every frame is in alpha-numerical order
        """
        # Load frame names
        frames = os.listdir(in_dir)

        if out_dir is None:
            out_dir = in_dir.strip("/").strip("\\") + "_yolo"

        # Check and make output directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if save_as != "jpg":
            # (x1, y1) = top-left corner
            # (x2, y2) = bottom-right corner
            temp_output = [["image", "x1", "y1", "x2", "y2", "confidence"]]

        for f in frames:
            frame = cv2.imread(os.path.join(in_dir, f))
            if rescale is not None:
                frame = cv2.resize(frame, rescale)
            pred = self.predict(frame, conf_thres=conf_thres, iou_thres=iou_thres)

            if save_as!="csv":
                self.draw(pred, frame, show=False)
                cv2.imwrite(os.path.join(out_dir, f), frame)

            if save_as!="jpg":
                d = self.extract_bounding_box_data(pred).tolist()
                temp_output = temp_output + [[f] + det for det in d]


        if save_as !="jpg":
            with open(os.path.join(out_dir, "bounding_boxes.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerows(temp_output)


def parent_dir(back, d=None,):
    # Goes back to the parent directory 'back' times
    if d is None:
        d = os.getcwd()

    parent = os.path.abspath(os.path.join(d, os.pardir))
    if back > 1:
        return parent_dir(back-1, parent)
    else:
        return parent

if __name__ == "__main__":
    from PIL import Image

    manager = YoloManager(image_size=[640,640], device="0")


    img = cv2.imread(os.path.join("data", "custom", "raising_hand.jpg"))

    preprocessed_img = manager.preprocess_frame(img)
    pred = manager.predict(img, conf_thres=0.30, scale_to_original=True)
    box, point = manager.extract_bounding_box_and_keypoint(pred)

    # Visualising
    manager.draw(pred, img)
    """
    in_dir = os.path.join(parent_dir(3), "occlusion_test", "frames", "street")
    out_dir = os.path.join(parent_dir(3), "occlusion_test", "frames", "street_640p")

    manager.draw_and_save(in_dir, out_dir, rescale=(640,640), save_as="csv")
    """