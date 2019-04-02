import cv2
import numpy as np
import pickle
from utils.reading import read_annotations_file
from utils.detection import Detection
from typing import List
from block_matching import block_matching_optical_flow
from optical_flow import farneback
from evaluation.bbox_iou import detection_iou

# Groundtruth
video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"

# Given detections
detections_path = "../datasets/AICity_data/train/S03/c010/det/"
detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]
roi_path = '../datasets/AICity_data/train/S03/c010/roi.jpg'

# Own detections
mask_detections_path = "../annotations/Mask-RCNN-detections.txt"
yolo_detections_path = "../annotations/yolo_detections.txt"
yolo_fine_tuned_detections_path = "../annotations/yolo_fine_tunning_detections.txt"


def track_optical_flow_sequence():
    print("Getting groundtruth")
    groundtruth_list, tracks_gt_list = read_annotations_file(groundtruth_xml_path, video_path)
    print("\nGetting detections")
    detections_list, _ = read_annotations_file(mask_detections_path, video_path)
    capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame = None
    frame_old = None  # possibly unnecessary
    detections_on_old_frame = None
    detections_on_frame = None
    while capture.isOpened():
        frame_old = frame
        valid, frame = capture.read()
        detections_on_old_frame = detections_on_frame
        detections_on_frame = [x for x in detections_list if x.frame == frame_idx]
        track_optical_flow_frame(frame_old, frame, detections_on_old_frame, detections_on_frame)
    

def track_optical_flow_frame(frame1, frame2, detection1 : List[Detection], detection2 : List[Detection]):
    INTERSECTION_THRESHOLD = 0.7
    feature_params = dict(maxCorners=500,
                     qualityLevel=0.3,
                     minDistance=7,
                     blockSize=7)
    det1_flow = []
    mask = np.zeros((frame1.shape[0], frame1.shape[1]), dtype=np.uint8) + 255
    if frame1 is not None and frame2 is not None:
        for det in detection1:
            mask[det.xtl:det.xtl + det.width, det.ytl:det.ytl + det.height] = 0
        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), mask=mask, **feature_params)
        # block matching:
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, p0, 0.5, 3, 15, 3, 5, 1.2, 0)
        for det in detection1:
            det_flow = flow[det.xtl:det.xtl + det.width, det.ytl:det.ytl + det.height,:]
            accum_flow = np.mean(det_flow[np.logical_or(det_flow[:, :, 0] != 0, det_flow[:, :, 1] != 0)], axis=(0, 1))
            if np.isnan(accum_flow):
                accum_flow = (0, 0)
            det1_flow.append(
                Detection(det.frame, det.label, det.xtl, det.ytl, det.width, det.height))

        for det in detection2:
            for det_old in det1_flow:
                if detection_iou(det, det_old) > INTERSECTION_THRESHOLD:
                    det.track_id = det_old.track_id
            


