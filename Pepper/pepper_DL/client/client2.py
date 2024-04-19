import sys
import numpy as np
import os
import cv2
import time
import requests
import io
import base64
import traceback
from PIL import Image

# SORT:
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "sort"))
#from trackers.sort.sort import SortManager

# OCSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "ocsort"))
from trackers.ocsort.ocsort import OCSortManager

# BoTSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "botsort"))
from trackers.botsort.bot_sort import *

# BoTSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "botsort"))
from trackers.botsort.bot_sort import BoTSortManager, bot_sort_make_parser

# ByteTrack:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "bytetrack"))
from trackers.bytetrack.byte_tracker import ByteTrackManager, byte_track_make_parser


models = {
    #"sort":SortManager,
    "ocsort":OCSortManager,
    "botsort":BoTSortManager,
    "bytetrack":ByteTrackManager,
}


class Client:

    def __init__(self, model="ocsort", address='http://localhost:5000', verbose=False, experimental=False, walk_speed_modifier=0.9, visualise_detector = True, **kwargs):
        self.address = address
        self.robot_actions = {
            "walkToward": self.walkToward,
            "walkTo": self.walkTo,
            "rotate_head": self.rotate_head,
            "rotate_head_abs": self.rotate_head_abs,
            "say": self.say,
            "target_lost": self.target_lost,
            "target_detected": self.target_detected,
        }
        self.model_name = model
        self.verbose = verbose
        self.vertical_ratio = None
        self.horizontal_ratio = None
        self.last_box = None
        print(f"Loading {model}...")
        self.dl_model = models[model](**kwargs)
        print(model, " loaded successfully!")
        self.walk_speed_modifier = walk_speed_modifier
        self.tracking = False
        self.visualise_detector = visualise_detector
        self.experimental = experimental
        if experimental:
            self.start_time = 0
            self.end_time = None
            self.terminate = False

    def get_image(self, show=False, save=False, path=None, save_name=None):
        headers = {'content-type':"/image/send_image"}
        response = requests.post(self.address+headers["content-type"], headers=headers)
        j = response.json()
        img = np.array(Image.open(io.BytesIO(base64.b64decode(j['img']))))[:,:,::-1] # Convert from BGR to RGB
        if show: # Don't use if running remotely
            cv2.imshow("Pepper_Image", img)
            cv2.waitKey(1)
        if save:
            cv2.imwrite(f"images/{save_name}.png" if path is None else os.path.join(path, save_name), img)
        return img

    def predict(self, img, draw=True, save_dir = None, show=False):
        if img is None:
            img = self.get_image()
        # Shape of pred: number of tracked targets x 5
        # where 5 represents: (x1, y1, x2, y2, id)
        detector_output = self.dl_model.detector_predict(img)
        pred, kpts = self.dl_model.smart_update(img, detector_output)

        if draw:
            if not self.tracking and self.visualise_detector:
                self.dl_model.detector.draw(detector_output, np.ascontiguousarray(img), show=show)
            else:
                self.draw(pred, img, save_dir="images" if save_dir is None else save_dir, show=show, kpts=kpts)

        return pred, img

    def draw(self, prediction, img, show=None, save_dir=None, save=False, kpts=None):
        self.dl_model.draw(prediction, np.ascontiguousarray(img), show=show, save_dir=save_dir, kpts=kpts, draw_kpts=kpts is not None)

    def follow_behaviour(self, draw=False, show=False, spin_speed=0.1):
        self.stop()
        try:
            while True:
                self.rotate_head_abs(verbose=False)
                ctarget_id = self.dl_model.target_id
                #if self.dl_model.target_id != self.dl_model.max_target_id:
                #    self.spin(speed=spin_speed)
                pred, img = self.predict(img=None, draw=draw, show=show)
                #print("Prediction:", pred)
                if ctarget_id == 0:
                    if ctarget_id != self.dl_model.target_id :
                        self.stop()
                        self.tracking = True
                        self.say("Target detected")

                else:
                    if ctarget_id != self.dl_model.target_id:
                        self.stop()
                        self.tracking = False
                        self.say("Target Lost")
                #print("Length of pred: ", len(pred))
                self.center_target(pred, img.shape, )
                self.last_box = pred
        except Exception as e:
            #print(e)
            traceback.print_exc()
            self.shutdown()

    # Only use if experimental is True
    def experiment_follow(self, save_dir=None, draw=False):
        data = {
                    "frames":0,
                    "time":0,
                }
        self.stop()
        failure = False
        try:
            start_time = time.time()
            while True:
                self.rotate_head_abs(verbose=False)
                ctarget_id = self.dl_model.target_id
                st = time.time()
                pred, img = self.predict(img=None, draw=draw)
                data["time"] = data["time"] + time.time() - st
                data["frames"] = data["frames"] + 1
                if draw:
                    self.draw(pred, img, save_dir=save_dir)
                #print("Prediction shape:", pred.shape)
                #print("Image shape:", img.shape)
                if ctarget_id == 0:
                    if ctarget_id != self.dl_model.target_id:
                        self.stop()
                        self.say("Target detected")
                        self.start_time = time.time()

                else:
                    if ctarget_id != self.dl_model.target_id:
                        self.stop()
                        self.say("Target Lost")
                        failure = True
                        break

                self.center_target(pred, img.shape, experimental=True)
                if self.terminate and self.experimental:
                    break
                self.last_box = pred
        except Exception as e:
            print(e)
        finally:
            #self.shutdown()
            data["start_time"] = start_time
            data["behaviour_time"] = self.end_time
            data["occluded_frame_count"] = self.dl_model.largest_target_absent_frames
            data["failure"] = failure
            return data

    def center_target(self, box, img_shape, stop_threshold = 0.1, vertical_offset=0.5, experimental=False):
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
            if self.dl_model.target_id == 0:
                print("Target Lost")
                #self.target_lost()
            else:
                self.rotate_head_abs()
        elif len(box) == 0:
            # If the length of box is zero, that means Pepper just lost track of the target before it officially
            # declares the target lost. In this window, we can still recover the track by making Pepper move towards
            # wherever the target could've shifted to
            if self.vertical_ratio is not None and self.horizontal_ratio is not None and self.dl_model.target_id!=0:
                self.walkToward(theta=self.horizontal_ratio*1.5)

        else: # If there's only 1 track, center the camera on them
            # Since there's an extra dimension, we'll take the first element, which is just the single detection
            box = box[0]

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

            if abs(horizontal_ratio) > stop_threshold:
                # If horizontal ratio is not within the stop threshold, rotate to center the target
                self.walkToward(theta=horizontal_ratio*self.walk_speed_modifier)
            else:
                # Otherwise, approach target
                self.approach_target(box, img_shape, commands=["rotate_head"],commands_kwargs=[{"forward":vertical_ratio*0.2}])



    def approach_target(self, box, img_shape, stop_threshold=0.70, move_back_threshold=0.9, commands=None, commands_kwargs=None):
        # (x1, y1, x2, y2, id)
        box_area = (box[2]-box[0])*(box[3]-box[1])
        frame_area = img_shape[0]*img_shape[1]
        ratio = box_area/frame_area
        if ratio >= stop_threshold:
            # Add end condition here:
            self.stop()

            if self.experimental:
                self.end_time = time.time() - self.start_time
                self.terminate = True
                self.say("Trial Success")
            else:
                self.say("Hello, I'm Pepper, do you require my assistance?")
            self.dl_model.reset_trackers()
            """
            if ratio > move_back_threshold:
                self.walkTo(x=(ratio-1)/3)
            else:
                if commands is not None: # assumes that commands is a list
                    for i in range(len(commands)):
                        self.robot_actions[commands[i]](**commands_kwargs[i])
            """
        else:
            self.walkToward(x=1-ratio, y=self.horizontal_ratio*ratio)

    def spin(self, left=True, speed=0.2, verbose = False):
        self.walkToward(theta = speed if left else -speed, verbose=verbose)
        # Keep resetting head position
        self.rotate_head_abs()

    #-------------------------------------------------------------------------------------------------------------------
    # Robot controls ###################################################################################################
    #-------------------------------------------------------------------------------------------------------------------

    def say(self, word, verbose = False):
        headers = {'content-type': "/voice/say"}
        response = requests.post(self.address + headers["content-type"], data=word)
        if verbose ^ self.verbose: # XOR
            print(f"say(word={word})")

    def target_lost(self, verbose = False):
        headers = {'content-type': "/voice/targetLost"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        if verbose ^ self.verbose:
            print(f"target_lost()")

    def target_detected(self, verbose = False):
        headers = {'content-type': "/voice/targetDetected"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        if verbose ^ self.verbose:
            print(f"target_detected()")

    def stop(self, verbose = False):
        headers = {'content-type': "/locomotion/stop"}
        response = requests.post(self.address + headers["content-type"])
        if verbose ^ self.verbose:
            print(f"stop()")

    def walkTo(self, x=0, y=0, theta=0, verbose=False):
        headers = {'content-type': "/locomotion/walkTo"}
        response = requests.post(self.address + headers["content-type"] + f"?x={str(x)}&y={str(y)}&theta={str(theta)}&verbose={str(1 if verbose else 0)}")
        if verbose ^ self.verbose:
            print(f"walkTo(x={str(x)}, y={str(y)}, theta={str(theta)})")

    def walkToward(self, x=0, y=0, theta=0, verbose=False):
        headers = {'content-type': "/locomotion/walkToward"}
        response = requests.post(self.address + headers["content-type"] + f"?x={str(x)}&y={str(y)}&theta={str(theta)}&verbose={str(1 if verbose else 0)}")
        if verbose ^ self.verbose:
            print(f"walkToward(x={str(x)}, y={str(y)}, theta={str(theta)})")

    def rotate_head(self, forward=0, left=0, speed=0.2, verbose=False):
        headers = {'content-type': "/locomotion/rotateHead"}
        response = requests.post(self.address + headers[
            "content-type"] + f"?forward={str(forward)}&left={str(left)}&speed={str(speed)}")
        if verbose ^ self.verbose:
            print(f"rotate_head(forward={str(forward)}, left={str(left)}, speed={str(speed)})")

    def rotate_head_abs(self, forward=0, left=0, speed=0.2, verbose=False):
        headers = {'content-type': "/locomotion/rotateHeadAbs"}
        response = requests.post(self.address + headers[
            "content-type"] + f"?forward={str(forward)}&left={str(left)}&speed={str(speed)}")
        if verbose ^ self.verbose:
            print(f"rotate_head_abs(forward={str(forward)}, left={str(left)}, speed={str(speed)})")


    def shutdown(self, verbose=False):
        headers = {'content-type': "/setup/end"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        if verbose ^ self.verbose:
            print("shutdown()")

    #------------------------------------------------------------------------------------------------------------------
    # Test code #######################################################################################################
    #------------------------------------------------------------------------------------------------------------------

    def get_image_test(self, image_no=60):
        start_time = time.time()
        for i in range(image_no):
            self.get_image(save=True, save_name=str(i))
        end_time = time.time() - start_time
        print(f"It took {str(end_time)} seconds to receive {str(image_no)} images. This means we were able to receive images from Pepper to server to client at {str(image_no/end_time)} FPS!")

    def pepper_to_server_fps(self, ):
        headers = {'content-type':"/test/pepper_to_server_fps"}
        response = requests.post(self.address + headers["content-type"], headers=headers)
        j = response.json()
        print(
            f"It took {str(j['time'])} seconds to receive {str(j['frames'])} images. This means we were able to receive images from Pepper to server to client at {str(float(j['frames']) / float(j['time']))} FPS!")

    def get_image_pred_test(self, image_no=60):
        start_time = time.time()
        for i in range(image_no):
            pred, img = self.predict(None, draw=False)
            end_time = time.time() - start_time
        print(f"It took {str(end_time)} seconds to receive {str(image_no)} images. This means we were able to receive images from Pepper to server to client at {str(image_no/end_time)} FPS!")

def dummy_action():
    # does nothing
    pass

def quick_shutdown():
    headers = {'content-type': "/setup/end"}
    response = requests.post("http://localhost:5000" + headers["content-type"], headers=headers)

if __name__ == "__main__":

    #args = byte_track_make_parser().parse_args()

    #c = Client(model="bytetrack", device="cuda", verbose=True, args=args,
    #           hand_raise_frames_thresh=3)

    #c.experiment_follow()

    c = Client(model="ocsort", device="cuda", verbose=True,
               hand_raise_frames_thresh=3)

    # c.get_image(save=True, path="exp_img/distance/forward", save_name="1m")

    # Must call
    #c.shutdown()
