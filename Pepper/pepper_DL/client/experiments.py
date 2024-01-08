import csv

from client2 import *
import requests
import os
import argparse

# OCSORT:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "ocsort"))
from trackers.ocsort.ocsort import OCSortManager

def initiate_oc(experimental=False, verbose = True, device="cuda"):
    return Client(model="ocsort", image_size=[640, 640], device=device, verbose=verbose, hand_raise_frames_thresh=3, experimental=experimental,)

def initiate_bot(experimental=False, verbose=True, device="cuda"):
    # BoTSORT default params
    args = bot_sort_make_parser().parse_args()
    args.ablation = False
    args.mot20 = not args.fuse_score

    return Client(model="botsort", image_size=[640, 640], device=device, verbose=verbose, experimental=experimental, args=args,
               hand_raise_frames_thresh=3)

def initiate_byte(experimental=False, verbose=True, device="cuda"):
    args = byte_track_make_parser().parse_args()

    return Client(model="bytetrack", device=device, verbose=verbose, experimental=experimental, args=args,
               hand_raise_frames_thresh=3)


def oc_exp(draw = True, trial="distance", distance="1m", attempt_no=1, verbose=False, clear_img=False, clear_log=False):
    p = os.path.join("exp_img", "OCSORT", trial, distance, str(attempt_no))
    print("Save directory:", p)
    if not os.path.exists(p):
        os.makedirs(p)
    elif clear_img:
        print("Clearing old images...")
        for f in os.listdir(p):
            os.remove(os.path.join(p,f))
        print("Clearing old images successful!")
    # Used for both the distance and occlusion trial for OCSORT
    c = initiate_oc(experimental=True, verbose=verbose)

    # Main follow behaviour:
    data = c.experiment_follow(save_dir=p, draw=draw)

    # Write up data
    data_writer(trial, attempt_no, distance, "OCSORT", data, clear_log=clear_log)

    print("Frames sent:", data["frames"])
    print("Time from detection to end condition:", data["behaviour_time"])
    print("FPS:", data["frames"] /data["time"])

def bot_exp(draw = True, trial="distance", distance="1m", attempt_no=1, verbose=False, clear_img=False, clear_log=False):
    p = os.path.join("exp_img", "BoTSORT", trial, distance, str(attempt_no))
    if not os.path.exists(p):
        os.makedirs(p)
    elif clear_img:
        print("Clearing old images...")
        for f in os.listdir(p):
            os.remove(os.path.join(p, f))
        print("Clearing old images successful!")
    # Used for both the distance and occlusion trial for OCSORT
    c = initiate_bot(experimental=True, verbose=verbose)
    # Main follow behaviour:
    data = c.experiment_follow(save_dir=p, draw=draw)

    # Write up data
    data_writer(trial, attempt_no, distance, "BoTSORT", data, clear_log=clear_log)

    print("Frames sent:", data["frames"])
    print("Time from detection to end condition:", data["behaviour_time"])
    print("FPS:", data["frames"] /data["time"])


def byte_exp(draw = True, trial="distance", distance="1m", attempt_no=1, verbose=False, clear_img=False, clear_log=False):
    p = os.path.join("exp_img", "ByteTrack", trial, distance, str(attempt_no))
    if not os.path.exists(p):
        os.makedirs(p)
    elif clear_img:
        print("Clearing old images...")
        for f in os.listdir(p):
            os.remove(os.path.join(p, f))
        print("Clearing old images successful!")
    # Used for both the distance and occlusion trial for OCSORT
    c = initiate_byte(experimental=True, verbose=verbose)

    data = c.experiment_follow(save_dir=p, draw=draw)

    # Write up data
    data_writer(trial, attempt_no, distance, "BYTETRACK", data, clear_log=clear_log)

    print("Frames sent:", data["frames"])
    print("Time from detection to end condition:", data["behaviour_time"])
    print("FPS:", data["frames"] /data["time"])

def ocfollow(verbose = True, device="cuda"):
    c = initiate_oc(verbose = verbose, device=device)
    #c = Client(image_size=[640, 640], device="cpu", max_age=60, verbose=True)
    # Main follow behaviour:
    c.follow_behaviour()
    # Must call
    c.shutdown()


def botfollow():
    c = initiate_bot()
    # Main follow behaviour:
    c.follow_behaviour()
    # Must call
    c.shutdown()


def livestream_camera_botsort():
    c = initiate_bot()
    vertical_offset = 0.5
    try:
        while True:
            pred, img = c.predict(img=None, draw=False)
            if len(pred) > 0:
                box = pred[0]
                img_shape = img.shape
                box_center = np.array([box[2] / 2 + box[0] / 2, box[1] * (1 - vertical_offset) + box[
                    3] * vertical_offset])  # box[1]/2+box[3]/2])
                frame_center = np.array((img_shape[1] / 2, img_shape[0] / 2))
                # diff = box_center - frame_center
                diff = frame_center - box_center
                horizontal_ratio = diff[0] / img_shape[1]
                vertical_ratio = diff[1] / img_shape[0]
                area = (box[2]-box[0])*(box[3]-box[1])
                area_ratio = area/(img_shape[0]*img_shape[1])
                print("BoT Prediction:", pred)
                print("Area ratio:", area_ratio)
                print("horizontal_ratio:", horizontal_ratio)
                print("vertical_ratio:", vertical_ratio)

    except Exception as e:
        print(e)
        c.shutdown()

def livestream_camera_ocsort():
    c = initiate_oc()
    vertical_offset = 0.5
    try:
        while True:
            pred, img = c.predict(img=None, draw=False)
            if len(pred) > 0:
                box = pred[0]
                img_shape = img.shape
                box_center = np.array([box[2] / 2 + box[0] / 2, box[1] * (1 - vertical_offset) + box[
                    3] * vertical_offset])  # box[1]/2+box[3]/2])
                frame_center = np.array((img_shape[1] / 2, img_shape[0] / 2))
                # diff = box_center - frame_center
                diff = frame_center - box_center
                horizontal_ratio = diff[0] / img_shape[1]
                vertical_ratio = diff[1] / img_shape[0]
                area = (box[2]-box[0])*(box[3]-box[1])
                area_ratio = area/(img_shape[0]*img_shape[1])
                print("Prediction:", pred)
                print("Area ratio:", area_ratio)
                print("horizontal_ratio:", horizontal_ratio)
                print("vertical_ratio:", vertical_ratio)

    except Exception as e:
        print(e)
        c.shutdown()

def get_image_pred_test(client, image_no=60):
    start_time = time.time()
    for i in range(image_no):
        pred, img = client.predict(img=None, draw=True)
        #print(pred.shape)
        end_time = time.time() - start_time
    print(f"It took {str(end_time)} seconds to receive {str(image_no)} images and process them through YOLO-Pose + {client.model_name}. This means we were able to receive images from Pepper to server to client at {str(image_no/end_time)} FPS!")


def yolo_experiment():
    c = OCSortManager(use_byte=True)
    exp_dir = os.path.join("exp_img", "resolution_test")
    raw_img_dir = os.path.join(exp_dir, "raw")
    output_dir = os.path.join(exp_dir, "yolo_output")

    resolutions = os.listdir(raw_img_dir)
    types = os.listdir(os.path.join(raw_img_dir, resolutions[0]))
    for resolution in resolutions:
        for t in types:
            im_dir = os.path.join("exp_img", "resolution_test", "raw", resolution, t)
            for im in os.listdir(im_dir):
            # Load image
                img = cv2.imread(os.path.join(im_dir, im))
                pred = c.detector_predict(img)
                c.draw(prediction=pred, img=img)
                cv2.imwrite(os.path.join("exp_img", "resolution_test", "yolo_output", resolution, t, im), img)

def data_writer(trial, attempt_no, distance, model, data, clear_log=False):
    header = None if not clear_log else ["Attempt_no", "Model", "Distance", "Time_to_Target", "FPS",
                                         "Occluded_Frames_Count", "Outcome"]
    write_entry(os.path.join("exp_logs", trial + "_log.csv"),
                [attempt_no, model, distance, data["behaviour_time"] if data["behaviour_time"] is not None else "NA", data["frames"] / data["time"],
                 data["occluded_frame_count"], "Failure" if data["failure"] else "Success"], header, "a" if not clear_log else "w")


def write_entry(file_dir, data, headers=None, mode="a"):
    with open(file_dir, mode) as f:
            writer = csv.writer(f)
            if headers is not None:
                writer.writerow(headers)
            writer.writerow(data)

def quick_shutdown():
    headers = {'content-type': "/setup/end"}
    response = requests.post("http://localhost:5000" + headers["content-type"], headers=headers)

def experiment_args():
    # For boolean variables, only specify those you want to be True. Ignore for False
    parser = argparse.ArgumentParser("Pepper Trial Experiment")

    # Logging args
    parser.add_argument(
        "--trial",
        default="occlusion",
        type=str,
        help="Type of trial we're doing, either 'occlusion' or 'distance'",
    )
    parser.add_argument(
        "--distance",
        default="1.5m",
        type=str,
        help="Distance between target and robot")
    parser.add_argument(
        "--attempt_no",
        default=1,
        type=int,
        help="The n-th attempt of this trial")

    parser.add_argument(
        "--model",
        default="ocsort",
        type=str,
        help="Model used to run trial. Will be used for both logging and determining which model to use. Currently, you can choose between 'ocsort', 'botsort', and 'bytetrack'")

    # Local files
    parser.add_argument(
        "--clear_img",
        default=False,
        type=bool,
        help="Determines whether we should clear images of a particular trial before running")

    parser.add_argument(
        "--clear_log",
        default=False,
        type=bool,
        help="Determines whether we should clear the log of a particular trial before running")

    parser.add_argument(
        "--draw",
        default=False,
        type=bool,
        help="Determines whether we should draw and save bounding boxes of all frames during this particular experiment")

    # Etc
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="Determines whether or not to print out the robot's actions at each step")

    return parser

# Used to initiate experiment instance by name
name_to_model = {
    "ocsort":oc_exp,
    "botsort":bot_exp,
    "bytetrack":byte_exp
}

import yaml
if __name__ == "__main__":
    # Run experiments
    #with open("config.yaml", "r") as yaml_file:
    #    args = yaml.safe_load(yaml_file)
    #m = name_to_model[args["model"]]
    #args.pop("model")
    #m(**args)

    # Run deep learning behaviour for end-user
    ocfollow(device="cpu")
    #oc_exp