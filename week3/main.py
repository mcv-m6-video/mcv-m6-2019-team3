import os
import pickle
import numpy as np

from evaluation.evaluation_funcs import compute_mAP_track
from utils.reading import read_annotations_file
from evaluation.evaluation_funcs import compute_mAP
from object_tracking.tracking import track_objects
from utils.candidate_generation_window import plot_bboxes
from utils.reading import read_annotations_file

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

if __name__ == '__main__':
    use_pkl = True
    export_frames = False

    # Read groundtruth
    print("Getting groundtruth")
    if use_pkl and os.path.exists('groundtruth.pkl'):
        with open('groundtruth.pkl', 'rb') as p:
            print("Reading detections from groundtruth.pkl")
            groundtruth_list = pickle.load(p)
    else:
        groundtruth_list, tracks_gt_list = read_annotations_file(groundtruth_xml_path, video_path)
        with open('groundtruth.pkl', 'wb') as f:
            pickle.dump(groundtruth_list, f)

    # Read Mask-RCNN detections from Task 1
    print("\nGetting detections")
    detections_list = read_annotations_file(mask_detections_path, video_path)

    # Compute mAP
    print("\nComputing mAP")
    compute_mAP(groundtruth_list, detections_list)

    # Print bboxes
    #print("\nExporting frames")
    #plot_bboxes(video_path, groundtruth_list, detections_list, export_frames=export_frames)

    # Task 2
    tracks = track_objects(video_path, groundtruth_list, groundtruth_list)
    # compute_mAP_track(tracks_gt_list, tracks)

    #Read detections files
    for detector in detectors:
        print(detector)
        detections_list = read_annotations_file(detections_path + detector, video_path)
        tracks = track_objects(video_path, detections_list, groundtruth_list)