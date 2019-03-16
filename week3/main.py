import os
import pickle
import numpy as np

from utils.reading import read_annotations_file
from object_tracking.tracking import track_objects

video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "../annotations/m6-full_annotation.xml"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
roi_path = '../datasets/AICity_data/train/S03/c010/roi.jpg'
detections_path = "./datasets/AICity_data/train/S03/c010/det/"
detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]


if __name__ == '__main__':

    # Read groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations_file(groundtruth_xml_path, video_path)
    tracks = track_objects(video_path, groundtruth_list)

    #Read detections files
    for detector in detectors:
        print(detector)
        detections_list = read_annotations_file(detections_path + detector, video_path)
        tracks = track_objects(video_path, detections_list)