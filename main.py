import cv2
import numpy as np

import xml.etree.ElementTree as ET
from typing import Iterator, Tuple

from evaluation.evaluation_funcs import compute_IoU, performance_evaluation_window, compute_mAP
from utils.reading import read_annotations, read_detections, read_annotations_from_txt
from utils.modify_detections import obtain_modified_detections


video_path = "./datasets/AICity_data/train/S03/c010/vdo.avi"

#groundtruth_path = "./datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml"
#groundtruth_path = "./datasets/AICity_data/train/S03/c010/Anotation_740-1090_AICITY_S03_C010.xml"
#groundtruth_path = "./6_vdo.xml"
groundtruth_path = "./datasets/AICity_data/train/S03/c010/det/det_yolo3.txt"

detections_path = "./datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"

if __name__ == "__main__":

    # Get groundtruth
    print("Getting groundtruth")
    if (groundtruth_path.endswith('.txt')):
        groundtruth_list = read_annotations_from_txt(groundtruth_path)
    elif (groundtruth_path.endswith('.xml')):
        capture = cv2.VideoCapture(video_path)
        root = ET.parse(groundtruth_path).getroot()
        groundtruth_list, images = read_annotations(capture, root, 40)
    else:
        raise Exception('Incompatible filetype')

    # Get detections
    print("Getting detections")
    if (detections_path.endswith('.txt')):
        detections_list = read_annotations_from_txt(detections_path)
    elif (detections_path.endswith('.xml')):
        detections_list = read_detections(detections_path)
    else:
        raise Exception('Incompatible filetype')

    # Compute IoU
    print("\nComputing IoU")
    compute_IoU(video_path, groundtruth_list, detections_list)

    # Repeat with modified detections
    print("Computing IoU with modified detections")
    detections_modified = obtain_modified_detections(detections_list)
    compute_IoU(video_path, groundtruth_list, detections_modified)

    #Compute mAP
    print("\nComputing mAP")
    compute_mAP(groundtruth_list, detections_list)
