"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
from skimage import io

def parent_dir(back, d=None,):
  # Goes back to the parent directory 'back' times
  if d is None:
    d = os.getcwd()

  parent = os.path.abspath(os.path.join(d, os.pardir))
  if back > 1:
    return parent_dir(back-1, parent)
  else:
    return parent

# Adding Yolo path:
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "detection_models", "edgeai_yolov5"))
sys.path.insert(0, os.path.join(parent_dir(1,os.path.dirname(os.path.realpath(__file__))), "detection_models", "edgeai_yolov5"))

import cv2
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

# To run SORT as a standalone file, remove the dots at the front from the following 2 statements:
from ..detection_models.edgeai_yolov5.utils.plots import colors, plot_one_box
from ..detection_models.edgeai_yolov5.yolo import YoloManager



#from detection_models.edgeai_yolov5.yolo import YoloManager

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1

    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

class SortManager(Sort):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, conf_threshold=0.25, **kwargs):
    # weights='yoloposes_640_lite.pt', device="cpu", save_txt_tidl=True, image_size=[640,640], kpt_label=True
    Sort.__init__(self, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    self.conf_threshold=conf_threshold
    self.detector = YoloManager(**kwargs)

  def detector_predict(self, frame, augment=False, classes=None, agnostic_nms=False):
    return self.detector.predict(frame, augment=augment, conf_thres=self.conf_threshold, classes=classes, iou_thres=self.iou_threshold, agnostic_nms=agnostic_nms)

  def update(self, frame, pred = None, augment=False, classes=None, agnostic_nms=False, get_bb=True):
    if pred is None:
      pred = self.detector_predict(frame, augment=augment, classes=classes, agnostic_nms=agnostic_nms)
    if get_bb:
      # Get bounding boxes, if True, assume that pred is the output of detection model
        # otherwise, assume that pred is bounding box data
      bounding_boxes = self.detector.extract_bounding_box_data(pred)
    else:
      bounding_boxes = pred
    return super().update(np.asarray(bounding_boxes))

  def draw(self, prediction, img, show=None):
    for det_index, (*xyxy, id) in enumerate(reversed(prediction[:,:6])):
      plot_one_box(xyxy, img, label=(f'id: {str(int(id))}'), color=colors(0,True), line_thickness=2, kpt_label=False, steps=3, orig_shape=img.shape[:2])
    if show is not None:
      cv2.imshow("Image", img)
      cv2.waitKey(show)

  def draw_and_save(self, in_dir, out_dir=None, rescale=None, detection=None):
    """ Runs sort on every single frame in in_dir and saves it to out_dir
    Params:
      in_dir - directory of images we wish to run tracking on
      out_dir - directory to save images with tracked bounding boxes. If None, makes a new file in the parent dir with
        the same name as in_dir but with added '_sort' in the end
      rescale - a tuple of length 2 where the first element details the height in pixels and the second element details
        the width in pixels
      detection - the full directory of a csv file that contains information about bounding box detections
    Requires: every frame is in alpha-numerical order
    """

    # Load frame names
    frames = os.listdir(in_dir)

    if out_dir is None:
      out_dir = in_dir.strip("/").strip("\\") + "_sort"

    # Check and make output directory
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    if detection is not None:
      # If we do have bounding box, we load up detections as a DataFrame
      detections = pd.read_csv(detection)


      # If detection is None, we have no bounding box data, so we have to predict it from here
    for f in frames:
      frame = cv2.imread(os.path.join(in_dir, f))
      if rescale is not None:
        frame = cv2.resize(frame, rescale)

      if detection is not None:
        # 2 cases here: either no detection or detection
        cdet = detections["image"]==f
        if cdet.any(): # If this is true, then there's at least one detection
          # Grab detections with the same image name as current frame excluding the name column
          dets = detections[cdet].drop("image", axis=1)
        else: # otherwise, there is no detection, and we have to put in a placeholder
          dets = np.empty((0, 5))
        get_bb = False

      else:
        dets = None
        get_bb = True

      sort_box = self.update(frame, pred = dets, get_bb = get_bb)
      self.draw(sort_box, frame)
      cv2.imwrite(os.path.join(out_dir, f), frame)


if __name__ == '__main__':
  s = SortManager(conf_threshold=0.15)

  data_dir = os.path.join("detection_models", "edgeai_yolov5", "data", "photos")
  #data_dir = os.path.join("edgeai_yolov5", "data", "custom")

  frame1 = cv2.imread(os.path.join(data_dir, "frame1.jpg"))
  frame2 = cv2.imread(os.path.join(data_dir, "frame2.jpg"))
  #frame1 = cv2.imread(os.path.join(data_dir, "paris.jpg"))

  f1 = s.update(frame1)
  s.draw(f1, frame1, 0)
  """



  in_dir = os.path.join(parent_dir(2), "occlusion_test", "frames", "street") #os.path.join(parent_dir(2), "occlusion_test", "frames", "long_walk")
  out_dir = None #os.path.join(parent_dir(2), "occlusion_test", "frames", "long_walk_640p")
  detection_dir = os.path.join(parent_dir(2), "occlusion_test", "frames", "street_640p", "bounding_boxes.csv")
  s.draw_and_save(in_dir, out_dir, rescale=(640, 640), detection=detection_dir)
  """