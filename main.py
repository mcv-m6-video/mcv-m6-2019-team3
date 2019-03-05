from evaluation.evaluation_funcs import compute_IoU, performance_evaluation_window
import numpy as np
import xml.etree.ElementTree as ET
from typing import Iterator, Tuple
import cv2
from utils.reading import read_annotations, read_detections, read_annotations_from_txt

video_path = "./datasets/AICity_data/train/S03/c010/vdo.avi"

#groundtruth_path = "./datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml"
#groundtruth_path = "./datasets/AICity_data/train/S03/c010/Anotation_740-1090_AICITY_S03_C010.xml"
#groundtruth_path = "./6_vdo.xml"
groundtruth_path = "./datasets/AICity_data/train/S03/c010/det/det_yolo3.txt"

detections_path = "./datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"

if __name__ == "__main__":

    # Get groundtruth
    print("Getting groundtruth")
    groundtruth_list = read_annotations_from_txt(groundtruth_path)

    # Get detections
    print("Getting detections")
    detections_list = read_annotations_from_txt(detections_path)

    print("Computing IoU")
    compute_IoU(video_path, groundtruth_list, detections_list)




    #capture = cv2.VideoCapture(video_path)
    #root = ET.parse(annotation_path).getroot()

    #groundtruth_annotations, images = read_annotations(capture, root, 2200)
    #detections = read_detections(detections_path)
    #print(groundtruth_annotations)
    #print(detections)

    #for n, image in enumerate(images):
        #cv2.imwrite('frames/{:04d}.jpg'.format(n), image)
        #print(n)



    # # Sort by confidence
    # detections_list.sort(key=lambda x: x.confidence, reverse=True)






