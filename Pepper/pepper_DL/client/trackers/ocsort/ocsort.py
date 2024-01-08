"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function


import os
import sys
import cv2
import numpy as np

# Get rid of the dot at the front to use this as a standalone file.
from .association import *
#from association import *

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


from yolo import YoloManager
from utils.plots import colors, plot_one_box

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


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
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          # Remove dot at front to run as standalone
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          #from kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class OCSort(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

    def update(self, output_results):
        """
        Params:
          output_results - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

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


        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])

        last_boxes = np.array([trk.last_observation for trk in self.trackers])

        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


class OCSortManager(OCSort):
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3, conf_thresh=0.4, delta_t=3, asso_func="iou",
                 inertia=0.2, use_byte=False, reset_target_thresh=15, hand_raise_frames_thresh=3 , **kwargs):
        """ OCSortManager, does tracking and detection
        Params:
            reset_target_thresh: int
                The threshold for the number of frames where the target is not present before resetting target
        """
        OCSort.__init__(self, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, det_thresh=conf_thresh,
                        delta_t=delta_t, asso_func=asso_func, inertia=inertia, use_byte=use_byte)
        self.detector = YoloManager(**kwargs)
        self.target_id = 0
        self.max_target_id = 1
        self.target_absent_frames = 0
        self.largest_target_absent_frames = 0
        self.reset_target_thresh=reset_target_thresh
        self.save_frame_count = 0
        self.last_box = None
        self.hand_raise_frames_thresh = hand_raise_frames_thresh
        self.hand_raise_frames = 0

    def detector_predict(self, frame, augment=False, classes=None, agnostic_nms=False):
        return self.detector.predict(frame, augment=augment, conf_thres=self.det_thresh, classes=classes, iou_thres=self.iou_threshold, agnostic_nms=agnostic_nms)

    def update(self, frame, pred = None, augment=False, classes=None, agnostic_nms=False, target_only = False):
        if pred is None:
            pred = self.detector_predict(frame, augment=augment, classes=classes, agnostic_nms=agnostic_nms)
        bounding_boxes = self.detector.extract_bounding_box_data(pred)

        track = super().update(np.asarray(bounding_boxes))
        return track[track[:,-1]==self.target_id] if target_only else track

    def filtered_update(self, frame, augment=False, classes=None, agnostic_nms=False, kpt_conf_thresh=0.5):
        """ Before running self.update, checks to see if anyone's raising their hand
        Params:
            frame: array
                an array representing an input image that we want to run detection through
            ktp_conf_thresh: a float between 0 and 1
                determines the minimum confidence of relevant keypoints before they are considered for a keypoint height
                check during the filter process
        Returns:
            an array of tracking targets. Should only contain information about 1 target.
            However, in rare circumstances, multiple targets may be tracked. In this case,
            we will track the target with the highest overall confidence
            Otherwise, None
        """
        pred = self.detector_predict(frame, augment=augment, classes=classes, agnostic_nms=agnostic_nms)
        points = self.detector.extract_keypoint_data(pred)
        # Filters prediction by only keeping those with their wrist keypoint above the shoulder keypoint only if
        # both keypoints are have high confidence (visibility)
        pred = pred[((points[:,5,1]>points[:,9,1]) & ((points[:,5,2]>kpt_conf_thresh) & (points[:,9,2]>kpt_conf_thresh))) | ((points[:,6,1]>points[:,10,1]) & ((points[:,6,2]>kpt_conf_thresh) & (points[:,10,2]>kpt_conf_thresh)))]

        # If there are more than 1 preds after filtering, we know that Pepper is seeing multiple people with their
        # hands raised. In that case, we'll take the person with the highest confidence
        # An alternative is to take the person with the largest bounding box area
        if len(pred) >= 1:
            # Update hand raise frame
            self.hand_raise_frames += 1
            # By confidence
            pred = max(pred, key = lambda x: x[4]).unsqueeze(0)
            # By area
            #pred = max(pred, key = lambda x: (x[2]-x[0])*(x[3]-x[1])).unsqueeze(0)

            if self.hand_raise_frames >= self.hand_raise_frames_thresh:
                track = self.update(frame, pred=pred)
                #print("Tracked track: ", track)
                #self.hand_raise_frames = 0
            else:
                track = super().update(np.empty((0, 5)))

            #track = self.update(frame, pred=pred)

        else:
            # Hand raise frames must be consecutive
            self.hand_raise_frames = 0
            track = super().update(np.empty((0, 5)))  # Should I run tracking even though no target has been detected?

        if len(track) > 0:
            if self.target_id != int(track[0,-1]):
                self.target_id = int(track[0,-1])
                if self.target_id > self.max_target_id:
                    self.max_target_id = self.target_id
        #print("self.hand_raise_frames:", self.hand_raise_frames)
        return track

    def smart_update(self, frame, pred = None, augment=False, classes=None, agnostic_nms=False):
        #Made to be called by the client, automatically determines whether to call filtered_update or update
        if self.target_id <= 0: # When there's no tracked target

            out = self.filtered_update(frame=frame, augment=augment, classes=classes, agnostic_nms=agnostic_nms)
            #print("m", m)
        else: # When there's a tracked target
            out = self.update(frame=frame, pred=pred, augment=augment, classes=classes, agnostic_nms=agnostic_nms, target_only=True)
            if len(out) == 0: # When the tracked target is not present on the screen
                self.target_absent_frames += 1
                self.largest_target_absent_frames = self.target_absent_frames if self.largest_target_absent_frames < self.target_absent_frames else self.largest_target_absent_frames
                #print("Absent frames = ", self.target_absent_frames)
                if self.target_absent_frames >= self.reset_target_thresh: # If the number of frames where the target
                    # is absent is greater than or equal to the threshold, reset the target ID
                    print("Target ", self.target_id, " is missing, looking for new target.")
                    self.target_id = 0
                    self.target_absent_frames = 0

        if len(out) == 1 and self.target_id > 0:
            # Saves last track
            self.last_box = out

        #print("out shape = ", out.shape)
        #print("out = ", out)
        #print("target Id = ", self.target_id)
        #print("max target Id = ", self.max_target_id)

        return out

    def draw(self, prediction, img, show=None, save_dir = None):
        #if len(prediction) !=
        for det_index, (*xyxy, id) in enumerate(reversed(prediction[:,:6])):
            plot_one_box(xyxy, img, label=(f'id: {str(int(id))}'), color=colors(int(id),True), line_thickness=2, kpt_label=False, steps=3, orig_shape=img.shape[:2])
        if save_dir is not None:
            self.save_frame_count += 1
            file_name = os.path.join(save_dir, "{:08d}.jpg".format(self.save_frame_count))
            cv2.imwrite(file_name, img)
        if show is not None:
            cv2.imshow("Image", img)
            cv2.waitKey(show)

    def reset_trackers(self):
        self.trackers = []
        self.target_id = 0
        self.target_absent_frames = 0
        self.last_box = None

if __name__ == '__main__':

    def parent_dir(back, d=None,):
        # Goes back to the parent directory 'back' times
        if d is None:
            d = os.getcwd()

        parent = os.path.abspath(os.path.join(d, os.pardir))
        if back > 1:
            return parent_dir(back-1, parent)
        else:
            return parent

    oc = OCSortManager(use_byte=True)
    #data_dir = os.path.join(parent_dir(1), "detection_models", "edgeai_yolov5", "data", "photos")
    #data_dir = os.path.join("edgeai_yolov5", "data", "custom")
    #frame1 = cv2.imread(os.path.join(data_dir, "frame1.jpg"))
    #frame2 = cv2.imread(os.path.join(data_dir, "frame2.jpg"))
    #frame1 = cv2.imread(os.path.join(data_dir, "paris.jpg"))

    #f1 = oc.update(frame1)
    #oc.draw(f1, frame1, 0, save_dir=os.path.join(parent_dir(2), "pepper_test"))

    data_dir = os.path.join(parent_dir(1), "botsort", "imgs")
    output_dir = os.path.join(parent_dir(1), "botsort", "output")
    images = [os.path.join(data_dir, img) for img in sorted(os.listdir(data_dir))]
    for p in images:
        print(p)
        img = cv2.imread(p)
        output = oc.update(img)
        print(output)