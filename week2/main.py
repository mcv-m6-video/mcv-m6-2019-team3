import numpy as np

video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
detections_path = "../datasets/AICity_data/train/S03/c010/det/"
detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]

annotated_groundtruth = "./annotations/Anotation_740-1090_AICITY_S03_C010.xml"

if __name__ == "__main__":

    # Read video
    print("Loading video")
    