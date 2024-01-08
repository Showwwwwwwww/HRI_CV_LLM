import numpy as np
from collections import deque

import os
import sys
import argparse
import cv2
import os.path as osp

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

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState


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

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def box_id(self):
        """ [left, top, right, bottom, id]
        """
        return np.append(self.tlbr, [self.track_id])

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


class ByteTrackManager(BYTETracker):
    def __init__(self, args, conf_thresh=0.4, reset_target_thresh=15, hand_raise_frames_thresh=3 , **kwargs):
        """ OCSortManager, does tracking and detection
        Params:
            reset_target_thresh: int
                The threshold for the number of frames where the target is not present before resetting target
        """
        BYTETracker.__init__(self, args, args.fps)
        self.detector = YoloManager(**kwargs)
        self.iou_threshold = args.iou_threshold
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
        return self.detector.predict(frame, augment=augment, conf_thres=self.args.track_thresh, classes=classes, iou_thres=self.iou_threshold, agnostic_nms=agnostic_nms)

    def update(self, frame, pred = None, augment=False, classes=None, agnostic_nms=False, target_only = False):
        if pred is None:
            pred = self.detector_predict(frame, augment=augment, classes=classes, agnostic_nms=agnostic_nms)
        bounding_boxes = self.detector.extract_bounding_box_data(pred)
        #print("YOLO prediction:", bounding_boxes)
        track = super().update(np.asarray(bounding_boxes), frame.shape, frame.shape)
        track = np.array([t.box_id for t in track if t.track_id == self.target_id]) if target_only else np.array(
            [t.box_id for t in track])
        return track if len(track) > 0 else np.empty((0, 5))

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
        #print("YOLO prediction:", pred)
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
                track = np.empty((0, 5))
            #track = self.update(frame, pred=pred)

        else:
            # Hand raise frames must be consecutive
            self.hand_raise_frames = 0
            track = np.empty((0, 5))  # Should I run tracking even though no target has been detected?

        if len(track) > 0:
            if self.target_id != int(track[0, -1]):
                self.target_id = int(track[0, -1])
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
        #return np.array([np.array(t.box_id) for t in out]) # Need to convert to [x1, y1, x2, y2, id] format

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
        self.removed_stracks.extend(self.tracked_stracks)
        self.tracked_stracks = []
        self.target_id = 0
        self.target_absent_frames = 0
        self.last_box = None


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def byte_track_make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # Detection Model params
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='iou threshold for detections. Detections with lower iou will be ignored')

    return parser
