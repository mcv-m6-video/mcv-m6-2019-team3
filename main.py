import cv2
import numpy as np

import xml.etree.ElementTree as ET
from typing import Iterator, Tuple

from evaluation.evaluation_funcs import compute_IoU, performance_evaluation_window, compute_mAP
from utils.reading import read_annotations, read_detections, read_annotations_from_txt, read_annotations
from utils.modify_detections import obtain_modified_detections
from evaluation.temporal_analysis import plotIoU, plotF1


video_path = "./datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_path = "./datasets/AICity_data/train/S03/c010/gt/gt.txt"

detections_path = "./datasets/AICity_data/train/S03/c010/det/"
detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]

#groundtruth_path = "./datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml"
#groundtruth_path = "./datasets/AICity_data/train/S03/c010/Anotation_740-1090_AICITY_S03_C010.xml"
#groundtruth_path = "./6_vdo.xml"
#groundtruth_path = "./datasets/AICity_data/train/S03/c010/det/det_yolo3.txt"

#detections_path = "./datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"

if __name__ == "__main__":

    # Get groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations(groundtruth_path)

    # Get detections
    print("Getting detections")
    detections_list = read_annotations(groundtruth_path)

    # Get modified detections
    detections_modified = obtain_modified_detections(detections_list)

    # Compute IoU
    print("\nComputing IoU")
    IoUFrames, F1Frames= compute_IoU(video_path, groundtruth_list, detections_list)
    plotIoU(IoUFrames, "./plots/IOUplots")
    plotF1(F1Frames, "./plots/F1plots")

    # Repeat with modified detections
    print("Computing IoU with modified detections")
    IoUFrames, F1Frames = compute_IoU(video_path, groundtruth_list, detections_modified)
    plotIoU(IoUFrames, "./plots/IOUplots_noise")
    plotF1(F1Frames, "./plots/F1plots_noise")

    #Compute mAP
    print("\nComputing mAP")
    compute_mAP(groundtruth_list, detections_list)

    # Repeat with modified detections
    print("Computing mAP with modified detections")
    compute_mAP(groundtruth_list, detections_modified)

    # Calculate mAP from detectors
    for detector in detectors:
        print(detector)
        detections_list = read_annotations(detections_path + detector)
        compute_mAP(groundtruth_list, detections_list)