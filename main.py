from evaluation.evaluation_funcs import performance_accumulation_pixel, performance_accumulation_window, performance_evaluation_pixel, performance_evaluation_window
import numpy as np
import xml.etree.ElementTree as ET
from typing import Iterator, Tuple
import cv2
from utils.reading import read_annotations, read_detections


video_path = "./datasets/AICity_data/train/S03/c010/vdo.avi"
annotation_path = "./datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml"
detections_path = "./datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"

if __name__ == "__main__":
    capture = cv2.VideoCapture(video_path)
    root = ET.parse(annotation_path).getroot()

    groundtruth_annotations, images = read_annotations(capture, root, 40)
    detections = read_detections(detections_path)
    print(groundtruth_annotations)
    print(detections)
